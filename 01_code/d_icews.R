sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

source("00_params.R")

libs <- c("tidyverse", "zoo", "lubridate", "countrycode")
ipak(libs)

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=F) %>% pull(.)

grab_reduced_icews <- FALSE
event_counts_path <- paste0(data_dir_base, "icews_counts.csv")

if (grab_reduced_icews==TRUE) {
  # helper to replace empty cells with NAs
  empty_as_na <- function(x) {
    ifelse(as.character(x)!="", x, NA)
  }
  
  reducedFiles <- list.files(icews_reduced_path)
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
  
  write_csv(counts, event_counts_path)
  
}

counts <- read_csv(event_counts_path)

counts$j_iso3 <- countrycode(counts$sourceCOW, "cown", "iso3c")
counts$i_iso3 <- countrycode(counts$tarCOW, "cown", "iso3c")

counts <- counts %>% filter(!is.na(j_iso3), !is.na(i_iso3))

#### SCALING ####

# see Archive/2016-2017/IR Event Data Project/   (ca wrapper?)