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

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE) %>% pull(.)
N <- length(ccodes)
year <- read_csv(paste0(dataPath, "year.csv"), col_names=FALSE) %>% pull(.)

X_star <- read_csv(paste0(estimatesPath, "X_star.csv"), col_names=FALSE) %>% as.matrix()
X_prime <- read_csv(paste0(estimatesPath, "X_prime.csv"), col_names=FALSE) %>% as.matrix()
X_diff <- X_prime - X_star

for (i in 1:N) {
  X_star[i, i] = NA
  X_prime[i, i] = NA
}

X_star_all <- sum(X_star, na.rm=T)
X_prime_all <- sum(X_prime, na.rm=T)

cftcl_trade_diff_frac <- 1 - X_prime_all / X_star_all

rownames(X_star) <- colnames(X_star) <- ccodes
X_star_melted <- melt(X_star) %>% as_tibble()
X_star_melted$X_star_log <- log(X_star_melted$value)
rownames(X_prime) <- colnames(X_prime) <- ccodes
X_prime_melted <- melt(X_prime) %>% as_tibble()
X_prime_melted$X_prime_log <- log(X_prime_melted$value)

# min_val <- min(c(X_star_melted$value, X_prime_melted$value), na.rm=T)
# max_val <- max(c(X_star_melted$value, X_prime_melted$value), na.rm=T)
  
# c_eqX_hm <- hm(X_star_melted, min_val, max_val, plot_title="Coercive Equilibrium: (Log) Trade Flows")
# nc_eqX_hm <- hm(X_prime_melted, min_val, max_val, plot_title="Non-Coercive Equilibrium: (Log) Trade Flows")

X_diff_melted <- left_join(X_star_melted, X_prime_melted, by=c("Var1", "Var2"))
X_diff_melted <- X_diff_melted %>% mutate(Var1 = factor(Var1, levels = rev(levels(Var1))))

min_val <- min(c(X_diff_melted$X_prime_log, X_diff_melted$X_star_log), na.rm=T)
max_val <- max(c(X_diff_melted$X_prime_log, X_diff_melted$X_star_log), na.rm=T)

point_size <- 1

X_diff_pp1 <- X_diff_melted %>% ggplot(aes(x=X_star_log, y=Var1, color="black")) + 
  geom_point(size=point_size) +
  geom_point(aes(x=X_prime_log, y=Var1, color="red"), alpha=0) +
  scale_x_continuous(limits=c(min_val, max_val)) +
  scale_color_manual("Equilibrium", values=c("black", "red"), labels=c("Coercive", "Non-Coercive"), guide="legend") +
  theme_classic() +
  theme(axis.ticks.x=element_blank(),
        axis.text.x=element_blank()) +
  labs(x="Trade Flows", y="Trade Partner", title=paste0("Effect of Coercion on (Log) Imports, ", year)) +
  facet_wrap(~Var2, nrow=2)

X_diff_pp2 <- X_diff_pp1 + 
  scale_x_continuous(limits=c(min_val, max_val)) +
  geom_point(aes(x=X_prime_log, y=Var1, color="red"), size=point_size) +
  geom_segment(aes(x=X_prime_log, xend=X_star_log, y=Var1, yend=Var1))
