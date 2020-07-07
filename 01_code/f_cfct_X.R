sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse", "reticulate", 'patchwork', 'reshape2')
ipak(libs)

use_virtualenv("python3")
c_setup <- import_from_path("c_setup", path=".")
setup <- c_setup$setup("local", "mid/")

ccodes <- read_csv(setup$ccodes_path, col_names=FALSE) %>% pull(.)
N <- length(ccodes)
year <- read_csv(paste0(setup$data_path, "year.csv"), col_names=FALSE) %>% pull(.)

X_star <- read_csv(setup$Xcif_path, col_names=FALSE) %>% as.matrix()

rownames(X_star) <- colnames(X_star) <- ccodes
X_star_melted <- melt(X_star) %>% as_tibble()
colnames(X_star_melted) <- c("i_iso3", "j_iso3", "X_star")
X_star_melted$X_star_log <- log(X_star_melted$X_star)

cfct_names <- c("cfct_demilitarization_path", "cfct_china_path", "cfct_us_path", "cfct_china_v_path")
cfct_X_L <- list()

for (i in cfct_names) {
  
  print(i)
  X_prime <- read_csv(paste0(setup[[i]], "X_prime.csv"), col_names=FALSE) %>% as.matrix()
  X_diff <- X_prime - X_star
  
  # for (j in 1:N) {
  #   X_star[j, j] = NA
  #   X_prime[j, j] = NA
  # }
  
  X_star_all <- sum(X_star, na.rm=T)
  X_prime_all <- sum(X_prime, na.rm=T)
  
  cftcl_trade_diff_frac <- 1 - X_prime_all / X_star_all
  
  rownames(X_prime) <- colnames(X_prime) <- ccodes
  X_prime_melted <- melt(X_prime) %>% as_tibble()
  colnames(X_prime_melted) <- c("i_iso3", "j_iso3", "X_prime")
  X_prime_melted$X_prime_log <- log(X_prime_melted$X_prime)
  
  X_diff_melted <- left_join(X_star_melted, X_prime_melted, by=c("i_iso3", "j_iso3"))
  X_diff_melted <- X_diff_melted %>% mutate(i_iso3 = factor(i_iso3, levels = rev(levels(i_iso3))))
  # X_diff_melted <- X_diff_melted %>% filter(Var1 != Var2)
  
  cfct_X_L[[i]] <- X_diff_melted
  
}

point_size <- 1

### COUNTERFACTUAL 1 ###

min_val <- min(c(cfct_X_L[["cfct_demilitarization_path"]]$X_prime_log, cfct_X_L[["cfct_demilitarization_path"]]$X_star_log), na.rm=T)
max_val <- max(c(cfct_X_L[["cfct_demilitarization_path"]]$X_prime_log, cfct_X_L[["cfct_demilitarization_path"]]$X_star_log), na.rm=T)

X_diff_pp_dm1 <- cfct_X_L[["cfct_demilitarization_path"]] %>% filter(i_iso3!=j_iso3) %>% ggplot(aes(x=X_star_log, y=i_iso3, color="black")) + 
  geom_point(size=point_size) +
  geom_point(aes(x=X_prime_log, y=i_iso3, color="red"), alpha=0) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c("black", "red"), labels=c("Coercive", "Non-Coercive"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Flows", y="Trade Partner", title="Effect of Coercion on International Trade", subtitle=paste0("Change in (Log) Imports")) +
  facet_wrap(~j_iso3, nrow=2)

X_diff_pp_dm2 <- X_diff_pp_dm1 + 
  scale_x_continuous(limits=c(min_val, max_val)) +
  geom_point(aes(x=X_prime_log, y=i_iso3, color="red"), size=point_size) +
  geom_segment(aes(x=X_prime_log, xend=X_star_log, y=i_iso3, yend=i_iso3))

ggsave(setup$f_cfact_demilitarization_Xprime_path)

### COUNTERFACTUAL 2 ###

X_diff_pp_china <- cfct_X_L[["cfct_china_path"]] %>% filter(i_iso3!=j_iso3) %>% ggplot(aes(x=X_star_log, y=i_iso3, color="black")) + 
  geom_point(size=point_size) +
  geom_point(aes(x=X_prime_log, y=i_iso3, color="red"), alpha=0) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c("black", "red"), labels=c("Baseline", "2030 Projected Military Capability"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Flows", y="Trade Partner", title="Effect of Multipolarization on International Trade", subtitle=paste0("Change in (Log) Imports, ", year)) +
  facet_wrap(~j_iso3, nrow=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  geom_point(aes(x=X_prime_log, y=i_iso3, color="red"), size=point_size) +
  geom_segment(aes(x=X_prime_log, xend=X_star_log, y=i_iso3, yend=i_iso3))

ggsave(setup$f_cfact_china_Xprime_path)

### COUNTERFACTUAL 3 ###

cfct_X_us <- cfct_X_L[["cfct_us_path"]]
cfct_X_us %>% group_by(i_iso3) %>% summarise(Y_star=sum(X_star), Y_prime=sum(X_prime)) 
cfct_X_us %>% filter(i_iso3 != j_iso3) %>% group_by(i_iso3) %>% summarise(X_sum_star=sum(X_star), X_sum_prime=sum(X_prime)) 

X_diff_pp_us <- cfct_X_L[["cfct_us_path"]] %>% filter(i_iso3!=j_iso3) %>% ggplot(aes(x=X_star_log, y=i_iso3, color="black")) + 
  geom_point(size=point_size) +
  geom_point(aes(x=X_prime_log, y=i_iso3, color="red"), alpha=0) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c("black", "red"), labels=c("Baseline", "U.S. Retrenchment"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Flows", y="Trade Partner", title="Effect of U.S. Retrenchment on International Trade", subtitle=paste0("Change in (Log) Imports, ", year)) +
  facet_wrap(~j_iso3, nrow=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  geom_point(aes(x=X_prime_log, y=i_iso3, color="red"), size=point_size) +
  geom_segment(aes(x=X_prime_log, xend=X_star_log, y=i_iso3, yend=i_iso3))

### COUNTERFACTUAL 4 ###

X_diff_pp_china_v <- cfct_X_L[["cfct_china_v_path"]] %>% filter(i_iso3!=j_iso3) %>% ggplot(aes(x=X_star_log, y=i_iso3, color="black")) + 
  geom_point(size=point_size) +
  geom_point(aes(x=X_prime_log, y=i_iso3, color="red"), alpha=0) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c("black", "red"), labels=c("Baseline", "Chinese Preference Liberalization"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Flows", y="Trade Partner", title="Effect of Chinese Preference Liberalization on International Trade", subtitle=paste0("Change in (Log) Imports, ", year)) +
  facet_wrap(~j_iso3, nrow=2) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  geom_point(aes(x=X_prime_log, y=i_iso3, color="red"), size=point_size) +
  geom_segment(aes(x=X_prime_log, xend=X_star_log, y=i_iso3, yend=i_iso3))
