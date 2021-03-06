# TODO: add MIDs, ICB
# TODO: counterfactual: welfare when others military spending is reduced

sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse", "zoo", "lubridate", "countrycode", "reticulate", "ca", "ggrepel", "utils", "readstata13")
ipak(libs)

use_virtualenv("python3")
c_setup <- import_from_path("c_setup", path=".")
setup <- c_setup$setup("local", "mid/")

ccodes <- read_csv(setup$ccodes_path, col_names=F) %>% pull(.)

year_L <- 2006
year_H <- 2016

#### EXTERNAL DATA ####

icb <- read_csv("http://people.duke.edu/~kcb38/ICB/icbdy_v12.csv") # statea/stateb are cow codes

mids_fname <- "dyadic mid 3.1_may 2018.dta"
temp <- tempfile()
download.file("https://correlatesofwar.org/data-sets/MIDs/dyadic-mids-and-dyadic-wars-v3.1/@@download/file/dyadic%20mids.zip", temp)
mids_f <- unzip(temp, files=c(mids_fname))
mids <- read.dta13(mids_f) %>% as_tibble()
file.remove(mids_fname)
unlink(temp)

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
  
  icews <- event.counts(events, 'year', 'quad')
  
  write_csv(icews, setup$icews_counts_path)
  
}

icews <- read_csv(setup$icews_counts_path)

### CLEAN ###

# i is source, j is target

mids <- mids %>% select(statea, stateb, year, hihost, rolea)
mids <- mids %>% filter(rolea=="Primary Initiator")  # filter to directed interations initiated by a

icb <- icb %>% select(statea, stateb, year)

mids$i_iso3 <- countrycode(mids$statea, "cown", "iso3c")
mids$j_iso3 <- countrycode(mids$stateb, "cown", "iso3c")
mids <- mids %>% select(-statea, -stateb)

icb$i_iso3 <- countrycode(icb$statea, "cown", "iso3c")
icb$j_iso3 <- countrycode(icb$stateb, "cown", "iso3c")
icb <- icb %>% select(-statea, -stateb)

icews$j_iso3 <- countrycode(icews$tarCOW, "cown", "iso3c")
icews$i_iso3 <- countrycode(icews$sourceCOW, "cown", "iso3c")  # i's events toward j, consistent with conquest vals (i conquering j)

# map EU
icews$j_iso3 <- mapEU(icews$j_iso3, icews$year)
icews$i_iso3 <- mapEU(icews$i_iso3, icews$year)
mids$j_iso3 <- mapEU(mids$j_iso3, mids$year)
mids$i_iso3 <- mapEU(mids$i_iso3, mids$year)
icb$j_iso3 <- mapEU(icb$j_iso3, icb$year)
icb$i_iso3 <- mapEU(icb$i_iso3, icb$year)

icews <- icews %>% filter(!is.na(j_iso3), !is.na(i_iso3)) %>% filter(i_iso3!=j_iso3) %>% select(i_iso3, j_iso3, year, everything()) %>% select(-sourceCOW, -tarCOW)
mids <- mids %>% filter(!is.na(j_iso3), !is.na(i_iso3)) %>% filter(i_iso3!=j_iso3) %>% select(i_iso3, j_iso3, year, everything())
icb <- icb %>% filter(!is.na(j_iso3), !is.na(i_iso3)) %>% filter(i_iso3!=j_iso3) %>% select(i_iso3, j_iso3, year, everything())
icb %>% arrange(desc(year)) %>% print(n=100)

icews <- icews %>% filter(year >= year_L, year <= year_H, i_iso3 %in% ccodes, j_iso3 %in% ccodes)
mids <- mids %>% filter(year >= year_L, year <= year_H, i_iso3 %in% ccodes, j_iso3 %in% ccodes)
icb <- icb %>% filter(year >= year_L, year <= year_H, i_iso3 %in% ccodes, j_iso3 %in% ccodes) # very few of these

# mids$i_iso3 <- as.factor(mids$i_iso3)
# mids$j_iso3 <- as.factor(mids$j_iso3)

# icews$era <- ntile(icews$year, 8)
# counts %>% filter(era==6) %>% pull(year) %>% unique()

# aggregate by era
# icews_era <- icews %>% group_by(i_iso3, j_iso3, era) %>% summarise(q1=sum(`1`), q2=sum(`2`), q3=sum(`3`), q4=sum(`4`))
# icews_era$n <- icews_era$q1 + icews_era$q2 + icews_era$q3 + icews_era$q4

icews_sub <- icews %>% 
  group_by(i_iso3, j_iso3) %>% summarise(q1=sum(`1`), q2=sum(`2`), q3=sum(`3`), q4=sum(`4`)) %>%
  mutate(n=q1+q2+q3+q4)
mids_sub <- mids %>% 
  group_by(i_iso3, j_iso3) %>% summarise(n_mids=n())

#### INTERNAL DATA ####

q_rcv <- read_csv(setup$quantiles_rcv_path, col_names=F)
pp <- read_csv(setup$quantiles_peace_probs_path, col_names=F)

quantiles_rcv <- expand.grid(ccodes, ccodes)
quantiles_rcv <- quantiles_rcv %>% cbind(q_rcv %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_rcv) <- c("i_iso3", "j_iso3", "rcv_q025", "rcv_q500", "rcv_q975")

quantiles_pp <- expand.grid(ccodes, ccodes)
quantiles_pp <- quantiles_pp %>% cbind(pp %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_pp) <- c("i_iso3", "j_iso3", "pp_q025", "pp_q500", "pp_q975")  # probability j faces attack from i

#### MODELS ####

counts_tpsp <- icews_sub %>% left_join(quantiles_rcv) %>% left_join(quantiles_pp)
counts_tpsp <- counts_tpsp %>% left_join(mids_sub)
counts_tpsp$n_mids <- ifelse(is.na(counts_tpsp$n_mids), 0, counts_tpsp$n_mids)
counts_tpsp <- counts_tpsp %>% arrange(j_iso3)

counts_tpsp$pp_inv <- 1 - counts_tpsp$pp_q500

icews_model_mc <- lm(data=counts_tpsp, q4 ~ pp_inv)  # material conflict
icews_model_mc_pos <- glm(formula = q4 ~ pp_inv, data=counts_tpsp, family="poisson")
summary(icews_model_mc)
summary(icews_model_mc_pos)

icews_model_vc <- lm(data=counts_tpsp, q3 ~ pp_inv)  # verbal conflict
icews_model_vc_pos <- glm(formula = q3 ~ pp_inv, data=counts_tpsp, family="poisson")
summary(icews_model_vc)
summary(icews_model_vc_pos)

# Note: we get similar results with regressions on cooperation, just picking up that EU and US interact more with others

mids_model <- lm(data=counts_tpsp, n_mids ~ pp_inv)  # mids (very few of these)
mids_model_pos <- glm(formula = n_mids ~ pp_inv, data=counts_tpsp, family="poisson")
summary(mids_model)
summary(mids_model_pos)

#### SCALING ICEWS ####

# Quad codes:
# 1 - verbal cooperation
# 2 - material cooperation
# 3 - verbal conflict
# 4 - material conflict

icews_sub_ca <- ca(icews_sub[3:6], nd=2)
# counts_sub_ca_1 <- ca(counts_sub[3:6], nd=1)
icews_sub$score1 <- icews_sub_ca$rowcoord[,1] %>% as.vector()
icews_sub$score2 <- icews_sub_ca$rowcoord[,2] %>% as.vector()
icews_scores <- icews_sub %>% select(i_iso3, j_iso3, score1, score2) %>% arrange(j_iso3)

counts_tpsp <- counts_tpsp %>% left_join(icews_scores)

icews_model_score <- lm(data=counts_tpsp, pp_inv ~ score2)
summary(icews_model_score)




### ARCHIVE ###


# counts_sub$score2 <- -1 * counts_sub$score2
# counts_sub$score1 <- -1 * counts_sub$score1
# high numbers consistent with more conflict in this run

counts_tpsp <- icews_sub

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

model1 <- lm(data=counts_tpsp, score1 ~ rcv_q500)
summary(model1)
model2 <- lm(data=counts_tpsp, score2 ~ rcv_q500)
summary(model2)
model3 <- lm(data=counts_tpsp, rcv_q500 ~ score1 + score2)
summary(model3)

ggplot(data=counts_tpsp, aes(x=score1, y=rcv_q500)) +
  geom_point() +
  geom_smooth(method="lm") +
  theme_classic()

ggplot(data=counts_tpsp, aes(x=score2, y=rcv_q500)) +
  geom_point() +
  theme_classic()
