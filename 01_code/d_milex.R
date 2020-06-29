sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse", "reticulate", "WDI", "reshape2")
ipak(libs)

use_virtualenv("python3")
c_setup <- import_from_path("c_setup", path=".")
setup <- c_setup$setup("local", "mid/")

ccodes <- read_csv(setup$ccodes_path, col_names=FALSE) %>% pull(.)
N <- length(ccodes)
year <- read_csv(paste0(setup$data_path, "year.csv"), col_names=FALSE) %>% pull(.)
years_proj <- seq(2019, 2030)

milex <- WDI(indicator="MS.MIL.XPND.CD", start=year, end=2018) %>% as_tibble()
milex$iso3c <- countrycode::countrycode(milex$iso2c, "iso2c", "iso3c")
milex$iso3c <- mapEU(milex$iso3c, milex$year)
milex <- milex %>% select(iso3c, MS.MIL.XPND.CD, year) %>% filter(iso3c %in% ccodes)
colnames(milex) <- c("iso3c", "milex", "year")
milex$milex <- milex$milex / 1000000000
milex <- milex %>% group_by(iso3c, year) %>% summarise(milex=sum(milex)) %>% ungroup()

models <- milex %>% group_by(iso3c) %>% do(model=lm(milex~year, data=.))

milex_f <- expand.grid(ccodes, years_proj) %>% as_tibble()
colnames(milex_f) <- c("iso3c", "year")
milex_f <- milex_f %>% arrange(iso3c)
milex_f$milex <- 0

for (i in ccodes) {
  if (i != "RoW") {
    model_i <- models$model[models$iso3c==i][[1]]
    preds <- predict(model_i, milex_f %>% filter(iso3c==i))
    milex_f$milex[milex_f$iso3c==i] <- preds
  }
}

milex <- bind_rows(milex, milex_f)

# milex %>% ggplot(aes(x=year, y=milex)) +
#   geom_line() +
#   theme_classic() +
#   facet_wrap(~iso3c)

M2030 <- milex %>% filter(year==2030) %>% arrange(iso3c)
M2030$milex[M2030$iso3c=="RoW"] <- min(M2030$milex[M2030$milex > 0])
M2030$milex <- M2030$milex / min(M2030$milex)

write_csv(M2030$milex %>% as.data.frame(), setup$M2030_path, col_names=F)