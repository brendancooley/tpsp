# TODO: add MIDs
# TODO: aggregate Europe, RoW for events

sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse", "zoo", "lubridate", "countrycode", "reticulate", "ca", "ggrepel")
ipak(libs)

use_virtualenv("python3")
c_setup <- import_from_path("c_setup", path=".")
setup <- c_setup$setup("local", "mid/")

ccodes <- read_csv(setup$ccodes_path, col_names=F) %>% pull(.)
q_rcv <- read_csv(setup$quantiles_rcv_path, col_names=F)
pp <- read_csv(setup$quantiles_peace_probs_path, col_names=F)

quantiles_rcv <- expand.grid(ccodes, ccodes)
quantiles_rcv <- quantiles_rcv %>% cbind(q_rcv %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_rcv) <- c("i_iso3", "j_iso3", "rcv_q025", "rcv_q500", "rcv_q975")

quantiles_pp <- expand.grid(ccodes, ccodes)
quantiles_pp <- quantiles_pp %>% cbind(pp %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_pp) <- c("i_iso3", "j_iso3", "pp_q025", "pp_q500", "pp_q975")  # probability j faces attack from i

grab_reduced_icews <- FALSE

if (grab_reduced_icews==TRUE) {
  # helper to replace empty cells with NAs
  empty_as_na <- function(x) {
    ifelse(as.character(x)!="", x, NA)
  }
  
  reducedFiles <- list.files(setup$icews_reduced_path)
  events.Y <- list()  # list holding data frames for each year
  # for each of the reduced files
  for (i in 1:length(reducedFiles)) {
    # append to list
    events.Y[[i]] <- read.delim(paste(icews_reduced_path, reducedFiles[i], sep=""), header = F)
    # convert column names
    colnames(events.Y[[i]]) <- c("date", "sourceName", "sourceCOW", "sourceSec",
                                 "tarName", "tarCOW", "tarSec", "CAMEO", "Goldstein", "quad")
    # replace empty cells with NAs
    events.Y[[i]] %>% mutate_if(is.factor, as.character) %>% mutate_all(funs(empty_as_na)) %>% as_tibble() -> events.Y[[i]]
  }
  # bind everything together
  events <- bind_rows(events.Y)
  
  # add year and month fields
  events$month <- as.yearmon(events$date)
  events$year <- year(events$date)
  events <- events %>% select(date, year, month, everything())
  
  event.counts <- function(events, agg.date=c('month', 'year'), code=c('quad', 'CAMEO')) {
    counts <- events %>%
      group_by_(agg.date, 'sourceCOW', 'tarCOW', code) %>%
      summarise(n = n()) %>%
      ungroup()  # this seems trivial but screws up a lot of stuff if you don't do it
    output <- spread_(counts, code, 'n')
    output[is.na(output)] <- 0
    return(output)
  }
  
  counts <- event.counts(events, 'year', 'quad')
  
  write_csv(counts, setup$icews_counts_path)
  
}

counts <- read_csv(setup$icews_counts_path)

counts$j_iso3 <- countrycode(counts$tarCOW, "cown", "iso3c")
counts$i_iso3 <- countrycode(counts$sourceCOW, "cown", "iso3c")  # i's events toward j, consistent with conquest vals (i conquering j)

counts <- counts %>% filter(!is.na(j_iso3), !is.na(i_iso3)) %>% filter(i_iso3!=j_iso3) %>% select(i_iso3, j_iso3, year, everything()) %>% select(-sourceCOW, -tarCOW)
counts$era <- ntile(counts$year, 8)
# counts %>% filter(era==6) %>% pull(year) %>% unique()

# aggregate by era
counts_era <- counts %>% group_by(i_iso3, j_iso3, era) %>% summarise(q1=sum(`1`), q2=sum(`2`), q3=sum(`3`), q4=sum(`4`))
counts_era$n <- counts_era$q1 + counts_era$q2 + counts_era$q3 + counts_era$q4

#### SCALING ####

# Quad codes:
# 1 - verbal cooperation
# 2 - material cooperation
# 3 - verbal conflict
# 4 - material conflict

# counts %>% filter(j_iso3=="USA", i_iso3=="CAN", year==1995)
# counts %>% filter(j_iso3=="CAN", i_iso3=="USA", year==1995)
# counts_era <- counts_era %>% filter(era==6, i_iso3 %in% ccodes, j_iso3 %in% ccodes)
counts_sub <- counts %>% filter(year >=2006, year <=2016, i_iso3 %in% ccodes, j_iso3 %in% ccodes) %>% 
  group_by(i_iso3, j_iso3) %>% summarise(q1=sum(`1`), q2=sum(`2`), q3=sum(`3`), q4=sum(`4`)) %>%
  mutate(n=q1+q2+q3+q4)

counts_sub_ca <- ca(counts_sub[3:6], nd=2)
counts_sub_ca_1 <- ca(counts_sub[3:6], nd=1)
counts_sub$score1 <- counts_sub_ca$rowcoord[,1] %>% as.vector()
counts_sub$score2 <- counts_sub_ca$rowcoord[,2] %>% as.vector()

# counts_sub$score2 <- -1 * counts_sub$score2
# counts_sub$score1 <- -1 * counts_sub$score1
# high numbers consistent with more conflict in this run

counts_tpsp <- counts_sub

counts_tpsp$ddyad <- paste0(counts_tpsp$i_iso3, "-", counts_tpsp$j_iso3)

ggplot(data=counts_tpsp, aes(x=score1, y=score2)) +
  geom_point() +
  geom_vline(xintercept=0, lty=2) +
  geom_hline(yintercept=0, lty=2) +
  geom_text_repel(aes(label=ddyad)) +
  theme_classic()

counts_tpsp <- counts_tpsp %>% left_join(quantiles_rcv) %>% left_join(quantiles_pp)
counts_tpsp %>% arrange(pp_q500) %>% print(n=100)
counts_tpsp$rcv <- counts_tpsp$rcv_q500 - 1
counts_tpsp$rcv_prime <- ifelse(counts_tpsp$rcv < 0, 0, counts_tpsp$rcv)

counts_tpsp$pp_inv <- 1 - counts_tpsp$pp_q500

counts_tpsp <- counts_tpsp %>% select(i_iso3, j_iso3, ddyad, q4, n, score1, score2, rcv_q500, pp_inv)
counts_tpsp %>% arrange(desc(q4))
counts_tpsp %>% arrange(desc(pp_inv))
quantiles_pp %>% arrange(pp_q500)

model1 <- lm(data=counts_tpsp, score1 ~ rcv_prime)
summary(model1)
model2 <- lm(data=counts_tpsp, score2 ~ rcv_prime)
summary(model2)
model3 <- lm(data=counts_tpsp, rcv_prime ~ score1 + score2)
summary(model3)

ggplot(data=counts_tpsp, aes(x=score1, y=rcv_prime)) +
  geom_point() +
  geom_smooth(method="lm") +
  theme_classic()

ggplot(data=counts_tpsp, aes(x=score2, y=rcv_prime)) +
  geom_point() +
  theme_classic()
