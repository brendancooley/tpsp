helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
helperFiles <- list.files(helperPath)
for (i in helperFiles) {
  source(paste0(helperPath, i))
}

libs <- c("tidyverse", "countrycode", "knitr", "kableExtra", "latex2exp")
ipak(libs)

tpspPath <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
dataPath <- paste0(tpspPath, "data/mid/")
resultsPath <- paste0(tpspPath, "results/mid/")
estimatesPath <- paste0(resultsPath, "estimates/")

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)
v_star <- read_csv(paste0(estimatesPath, "v.csv"), col_names=FALSE)

v_est_table_path <- "v_est_table.png"
ccodes_table_path <- "ccodes_table.png"
 
ccodesT <- ccodes
N <- nrow(ccodes)
colnames(ccodesT) <- c("iso3")
ccodesT$`Country Name` <- countrycode(ccodesT$iso3, "iso3c", "country.name")
ccodesT$`Country Name` <- ifelse(ccodesT$iso3=="EU", "European Union", ccodesT$`Country Name`)
ccodesT$`Country Name` <- ifelse(ccodesT$iso3=="RoW", "Rest of World", ccodesT$`Country Name`)

v_estsT <- cbind(ccodesT, round(v_star, 2))
colnames(v_estsT) <- c("iso3", "Country Name", "$\\tilde{v}_i$")

v_est_table <- kable(v_estsT, "latex", booktabs = T, caption = "Preference Parameter ($\\tilde{\\bm{v}}$) Estimates \\label{tab:v_estsT}", escape = FALSE) %>% kable_styling(position = "center", latex_options=c("striped"))
ccodes_table <- kable(ccodesT, "latex", booktabs = T, caption = "In-Sample Countries", escape = FALSE) %>% kable_styling(position = "center", latex_options=c("striped"))

save_kable(v_est_table, v_est_table_path)
save_kable(ccodes_table, ccodes_table_path)
