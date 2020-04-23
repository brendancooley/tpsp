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

# i's advantage over j
colnames(m_frac) <- ccodes
m_frac <- cbind(ccodes, m_frac) %>% as_tibble()
colnames(m_frac)[1] <- c("i_iso3")
m_frac_long <- m_frac %>% pivot_longer(-i_iso3, names_to="j_iso3", values_to="m_frac")

Y_j <- cbind(ccodes, Y)
colnames(Y_j) <- c("j_iso3", "Y_j")
Y_i <- cbind(ccodes, Y)
colnames(Y_i) <- c("i_iso3", "Y_i")

data <- tau_long %>% left_join(m_frac_long) %>% left_join(W_long) %>% left_join(Y_j) %>% left_join(Y_i)
data <- data %>% filter(j_iso3!=i_iso3)
data <- data %>% filter(j_iso3!="RoW" & i_iso3!="RoW")

data$m_frac_log <- log(data$m_frac)
data$Y_j_log <- log(data$Y_j)
data$Y_i_log <- log(data$Y_i)
data$W_log <- log(data$W)
data$tau_adv <- data$tau - 1
data$tau_adv_log <- log(data$tau_adv)

data %>% filter(j_iso3=="CHN")
data %>% print(n=100)

### VIZ ###

data %>% ggplot(aes(x=Y_j_log, y=tau)) +
  geom_point() +
  theme_classic()

data %>% ggplot(aes(x=m_frac_log, y=tau)) +
  geom_point() +
  geom_smooth(method="lm") + 
  theme_classic() + 
  facet_wrap(~j_iso3)

data %>% ggplot(aes(x=m_frac_log, y=tau_adv)) +
  geom_point() +
  theme_classic() + 
  facet_wrap(~i_iso3)

data %>% ggplot(aes(x=m_frac_log, y=tau, col=Y_j_log)) +
  geom_point() +
  theme_classic()

### REG ###

model_y <- lm(data=data, tau~Y_j_log)
summary(model_y)

model_my <- lm(data=data, tau~m_frac_log+Y_i_log)
summary(model_my)

model_myw <- lm(data=data, tau~m_frac_log*W_log+j_iso3)
summary(model_myw)

model_mfe <- lm(data=data, tau~i_iso3+j_iso3)
summary(model_mfe)
