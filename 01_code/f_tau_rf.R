sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse", "modelsummary", "dotwhisker", "ggsci", "reticulate", "broom", "patchwork", "kableExtra")
ipak(libs)

use_virtualenv("python3")
c_setup <- import_from_path("c_setup", path=".")
setup <- c_setup$setup("local", "mid/")

ccodes <- read_csv(setup$ccodes_path, col_names=F) %>% pull(.)
tau <- read_csv(setup$tau_path, col_names=F)
M <- read_csv(setup$M_path, col_names=F) %>% pull(.)
Y <- read_csv(setup$Y_path, col_names=F)
W <- read_csv(setup$dists_path, col_names=F)
X <- read_csv(setup$Xcif_path, col_names=F)
gdp <- read_csv(setup$gdp_raw_path)
rcv_ft <- read_csv(setup$rcv_ft_path, col_names=F) # %>% t() %>% as_tibble()

N <- length(ccodes)

colnames(tau) <- colnames(W) <- colnames(X) <- colnames(rcv_ft) <- ccodes

### RESHAPE ###

tau <- cbind(ccodes, tau)
colnames(tau)[1] <- c("j_iso3")
tau_long <- tau %>% pivot_longer(-j_iso3, names_to="i_iso3", values_to="tau")

W <- cbind(ccodes, W)
colnames(W)[1] <- c("j_iso3")
W_long <- W %>% pivot_longer(-j_iso3, names_to="i_iso3", values_to="W")

X <- cbind(ccodes, X)
colnames(X)[1] <- c("j_iso3")
X_long <- X %>% pivot_longer(-j_iso3, names_to="i_iso3", values_to="X_ji")

rcv_ft <- cbind(ccodes, rcv_ft)
colnames(rcv_ft)[1] <- c("j_iso3")
rcv_long <- rcv_ft %>% pivot_longer(-j_iso3, names_to="i_iso3", values_to="rcv_ij")  # i's value for attacking j
# rcv_long %>% arrange(desc(rcv_ij))

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
m_frac_long <- m_frac %>% pivot_longer(-i_iso3, names_to="j_iso3", values_to="m_frac_ij")

gdp_j <- cbind(ccodes, gdp)
colnames(gdp_j) <- c("j_iso3", "gdp_j")
gdp_i <- cbind(ccodes, gdp)
colnames(gdp_i) <- c("i_iso3", "gdp_i")

data <- tau_long %>% left_join(m_frac_long) %>% left_join(W_long) %>% left_join(gdp_j) %>% left_join(gdp_i) %>% left_join(X_long) %>% left_join(rcv_long)
data <- data %>% filter(j_iso3!=i_iso3)
data <- data %>% filter(j_iso3!="RoW" & i_iso3!="RoW")

### REG ###

# data$tau_rev_frac <- (data$tau - 1) * data$X_ji / data$gdp_j
# data$rcv_ij <- data$rcv_ij - 1
# data %>% filter(i_iso3 %in% c("USA", "EU")) %>% arrange(i_iso3, rcv_ij) %>% print(n=100)

model1 <- lm(data=data, log(rcv_ij**-1)~log(m_frac_ij))
model2 <- lm(data=data, log(rcv_ij**-1)~log(m_frac_ij)+i_iso3)  # attacker-specific costs w/ FE
model3 <- lm(data=data, log(rcv_ij**-1)~log(m_frac_ij)*log(W))  # conditional effect of distance
model4 <- lm(data=data, log(rcv_ij**-1)~log(m_frac_ij)*log(W)+i_iso3)  # attacker-specific costs w/ FE

# model5 <- lm(data=data, log(rcv_ij)~log(m_frac_ij)*log(W)+log(gdp_i))
# summary(model5)
# model6 <- lm(data=data, log(rcv_ij)~log(m_frac_ij)*log(W)+log(m_frac_ij)*log(gdp_i)+j_iso3)
# summary(model6)
# 
# model7 <- lm(data=data, log(rcv_ij)~log(m_frac_ij)+j_iso3)  # control for policy-chooser preferences
# summary(model7)

models <- list()
models[["Base"]] <- model1
models[["Base (Attacker FE)"]] <- model2
models[["Loss of Strength"]] <- model3
models[["Loss of Strength (Attacker FE)"]] <- model4

cm <- c("log(m_frac_ij)"="Log Mil Capability Ratio",
        "log(W)"="Log Distance",
        "log(m_frac_ij):log(W)"="(Log Mil Capability Ratio) X (Log Distance)")
fe_row <- c("Attacker FE?", " ", "\U2713", " ", "\U2713")

table_tex <- modelsummary(models, coef_map=cm, add_rows=list(fe_row), gof_omit="AIC|BIC|Log.Lik", title="Inverse Conquest Values and Military Capability Ratios", stars=TRUE, output="latex") %>% kable_styling(latex_options=c("scale_down"))
table_png <- modelsummary(models, coef_map=cm, add_rows=list(fe_row), gof_omit="AIC|BIC|Log.Lik", title="Inverse Conquest Values and Military Capability Ratios", stars=TRUE, output=setup$f_tau_rf_table_path)

for (i in names(models)) {
  models[[i]] <- models[[i]] %>% tidy() %>% filter(term %in% c("log(m_frac_ij)", "log(W)", "log(m_frac_ij):log(W)")) %>% mutate(model=i)
}

models <- bind_rows(models)
dw_colors <- kc_discrete_palette[1:4]

dw <- dwplot(models) %>%
  relabel_predictors(cm) +
  geom_vline(xintercept=0, lty=2) +
  scale_color_manual(values=dw_colors) +
  labs(color="Model", title="Correlates of Conquest Values", subtitle="Point estimates and 95 percent confidence intervals") +
  theme_classic()

ggsave(setup$f_tau_rf_dw_path, width=7, height=3.5)


### Raw Bivaritate Correlation 

rcvm_plot <- ggplot(data, aes(x=log(m_frac_ij), y=log(rcv_ij**-1))) +
  geom_point() +
  geom_smooth(method="lm", se=FALSE, color=bcOrange) +
  theme_classic() +
  labs(x="Military Capability Ratio (Log)", y="Inverse Conquest Value (Log)", title="Military Capability Ratios and Regime Change Values",
       subtitle="Observations: All In-Sample Directed Dyads") +
  theme(axis.text=element_blank(),
        axis.ticks=element_blank(),
        aspect.ratio=1)
  
