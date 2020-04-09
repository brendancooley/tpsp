helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
helperFiles <- list.files(helperPath)
for (i in helperFiles) {
  source(paste0(helperPath, i))
}

libs <- c("tidyverse", "patchwork", "reshape2")
ipak(libs)

tpspPath <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
dataPath <- paste0(tpspPath, "data/mid/")
resultsPath <- paste0(tpspPath, "results/mid/")
estimatesPath <- paste0(resultsPath, "estimates/")

gamma <- read_csv(paste0(estimatesPath, "gamma.csv"), col_names=FALSE) %>% pull(.)
alpha <- read_csv(paste0(estimatesPath, "alpha1.csv"), col_names=FALSE) %>% pull(.)
c_hat <- read_csv(paste0(estimatesPath, "c_hat.csv"), col_names=FALSE) %>% pull(.)

costf <- function(x, data, param, c_hat) {
  out <- exp(-c**(-1)*data^param*x^(-1))*c**(-1)*data^param*x^(-2)
  return(out)
}

costF <- function(x, data, param, c_hat) {
  out <- exp(-c**(-1)*data^param*x^(-1))
  return(out)
}

Mbase <- 1
Mprime <- 2

meanC_Mbase <- costF(.5, Mbase, gamma, c_hat) ** -1
meanC_Mprime <- costF(.5, Mprime, gamma, c_hat) ** -1

Wbase <- 1
Wprime <- 1000

meanC_Wbase <- costF(.5, Wbase, alpha, c_hat) ** -1
meanC_Wprime <- costF(.5, Wprime, alpha, c_hat) ** -1

x <- seq(.001, 5, .001)
costf_Mbase <- c()
costf_Mprime <- c()
costf_Wbase <- c()
costf_Wprime <- c()
for (i in x) {
  costf_Mbase_i <- costf(i, Mbase, gamma, c_hat)
  costf_Mprime_i <- costf(i, Mprime, gamma, c_hat)
  costf_Wbase_i <- costf(i, Wbase, alpha, c_hat)
  costf_Wprime_i <- costf(i, Wprime, alpha, c_hat)
  costf_Mbase <- c(costf_Mbase, costf_Mbase_i, c_hat)
  costf_Mprime <- c(costf_Mprime, costf_Mprime_i, c_hat)
  costf_Wbase <- c(costf_Wbase, costf_Wbase_i, c_hat)
  costf_Wprime <- c(costf_Wprime, costf_Wprime_i, c_hat)
}

costf_data <- data.frame(x, costf_Mbase, costf_Mprime, costf_Wbase, costf_Wprime) %>% as_tibble()
# costF_data

# ggplot(data=costF_data, aes(x=x)) +
#   geom_line(y=costF_Mbase) +
#   geom_line(y=costF_Mprime, color="red") +
#   ylim(0, max(costF_data$costF_Mbase)) +
#   theme_classic()
