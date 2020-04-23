sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse")
ipak(libs)

projectFiles <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
size <- "mid_RUS/"
data_dir_base <- paste0(projectFiles, "data/")
dataPath <- paste0(data_dir_base, size)

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=F) %>% pull(.)
N <- length(ccodes)
tau <- read_csv(paste0(dataPath, "tau.csv"), col_names=F)
M <- read_csv(paste0(dataPath, "milex.csv"), col_names=F) %>% pull(.)
Y <- read_csv(paste0(dataPath, "y.csv"), col_names=F)
W <- read_csv(paste0(dataPath, "cDists.csv"), col_names=F)

colnames(tau) <- colnames(W) <- ccodes

### RESHAPE ###

tau <- cbind(ccodes, tau)
colnames(tau)[1] <- c("j_iso3")
tau_long <- tau %>% pivot_longer(-j_iso3, names_to="i_iso3", values_to="tau")

W <- cbind(ccodes, W)
colnames(W)[1] <- c("j_iso3")
W_long <- W %>% pivot_longer(-j_iso3, names_to="i_iso3", values_to="W")

M <- M / min(M)
M_diag <- diag(M)
ones <- matrix(1, nrow=N, ncol=N)
m <- M / ones
m <- t(m)
m_frac <- t(m / M) %>% as.data.frame()

colnames(m_frac) <- ccodes
m_frac <- cbind(ccodes, m_frac) %>% as_tibble()
colnames(m_frac)[1] <- c("j_iso3")
m_frac_long <- m_frac %>% pivot_longer(-j_iso3, names_to="i_iso3", values_to="m_frac")

Y_j <- cbind(ccodes, Y)
colnames(Y_j) <- c("j_iso3", "Y_j")
Y_i <- cbind(ccodes, Y)
colnames(Y_i) <- c("i_iso3", "Y_i")

data <- tau_long %>% left_join(m_frac_long) %>% left_join(W_long) %>% left_join(Y_j) %>% left_join(Y_i)
data <- data %>% filter(j_iso3!=i_iso3)
data$m_frac_log <- log(data$m_frac)
data$Y_j_log <- log(data$Y_j)
data$Y_i_log <- log(data$Y_i)

### VIZ ###

data %>% ggplot(aes(x=m_frac_log, y=tau, col=j_iso3)) +
  geom_point() +
  theme_classic()

data %>% ggplot(aes(x=Y_j_log, y=tau, col=Y_i_log)) +
  geom_point() +
  theme_classic()

### REG ###

model_my <- lm(data=data, tau~m_frac_log+Y_j_log+Y_i_log)
summary(model_base)

model_mfe <- lm(data=data, tau~m_frac_log+j_iso3)
summary(model_mfe)
