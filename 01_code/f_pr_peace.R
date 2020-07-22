sourceDir <- paste0("../source/R/")
sourceFiles <- list.files(sourceDir)
for (i in sourceFiles) {
  source(paste0(sourceDir, i))
}

libs <- c("tidyverse", "reticulate", 'patchwork', 'reshape2', "ggrepel")
ipak(libs)

use_virtualenv("python3")
c_setup <- import_from_path("c_setup", path=".")
setup <- c_setup$setup("local", "mid/")

ccodes <- read_csv(setup$ccodes_path, col_names=FALSE) %>% pull(.)
N <- length(ccodes)
year <- read_csv(paste0(setup$data_path, "year.csv"), col_names=FALSE) %>% pull(.)

quantiles_peace_probs <- read_csv(setup$quantiles_peace_probs_path, col_names=FALSE)

quantiles_pp <- expand.grid(ccodes, ccodes)
quantiles_pp <- quantiles_pp %>% cbind(quantiles_peace_probs %>% t()) %>% as_tibble()  # i's value for conquering j
colnames(quantiles_pp) <- c("Var2", "Var1", "pp_q025", "pp_q500", "pp_q975")

# log(quantiles_pp$pp_q500)
quantiles_pp$value <- 1 - quantiles_pp$pp_q500
# quantiles_pp$value <- log(quantiles_pp$value)

# for (i in 1:N) {
#   war_probs[i, i] = NA
# }
# 
# rownames(war_probs) <- colnames(war_probs) <- ccodes
# war_probs_melted <- melt(war_probs) %>% as_tibble()
war_probs_melted <- quantiles_pp %>% select(Var1, Var2, value)

min_val <- min(war_probs_melted$value, na.rm=T)
max_val <- max(war_probs_melted$value, na.rm=T)

war_probs_hm <- hm(war_probs_melted, min_val, max_val, x="Attacker", y="Defender", plot_title="Point Estimates: Equilibrium War Probabilities")

ggsave(setup$f_pr_peace_path)

quantiles_pp2 <- quantiles_pp %>% filter(Var1 != Var2, Var1 !="RoW", Var2 != "RoW") %>% arrange(pp_q500)
quantiles_pp2$ddyad <- paste0(quantiles_pp2$Var2, "-", quantiles_pp2$Var1) %>% as.factor()
quantiles_pp2$ddyad <- fct_reorder(quantiles_pp2$ddyad, quantiles_pp2$pp_q500, .desc=TRUE)
quantiles_pp2$label <- ifelse(quantiles_pp2$pp_q500 < .5, as.character(quantiles_pp2$ddyad), "")
quantiles_pp2$USA <- ifelse(quantiles_pp2$Var2=="USA", bcOrange, "black")

war_probs_pp <- quantiles_pp2 %>% ggplot(aes(x=ddyad, y=1-pp_q500, label=label)) +
  geom_point(size=.5, color=quantiles_pp2$USA) + 
  geom_segment(aes(x=ddyad, xend=ddyad, y=1-pp_q025, yend=1-pp_q975), color=quantiles_pp2$USA) +
  geom_text_repel() +
  geom_hline(yintercept=0, lty=2) +
  theme_classic() +
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        aspect.ratio=1) +
  labs(x="Directed Dyad", y="Probability of War", title="Equilibrium Probabilities of War", subtitle="All directed dyads, point estimates and 95 percent confidence intervals")

### CHINESE LIBERALIZATION COUNTERFACTUAL ###

pp_prime4 <- read_csv(paste0(setup$cfct_china_v_path, "pp.csv"), col_names=F) %>% as.matrix() %>% t() %>% as.numeric()

quantiles_pp$pp_prime4 <- pp_prime4
quantiles_pp4 <- quantiles_pp %>% filter(Var1 != Var2, Var1 !="RoW", Var2 != "RoW") %>% arrange(pp_q500)
quantiles_pp4$ddyad <- paste0(quantiles_pp4$Var2, "-", quantiles_pp4$Var1) %>% as.factor()
quantiles_pp4$ddyad <- fct_reorder(quantiles_pp4$ddyad, quantiles_pp4$pp_q500, .desc=TRUE)
quantiles_pp4$Var2 <- fct_reorder(quantiles_pp4$Var2, quantiles_pp4$pp_q500, .desc=TRUE)

pw_USA <- 1 - quantiles_pp4 %>% filter(Var1=="CHN", Var2=="USA") %>% pull(pp_q500)
pw4_USA <- 1 - quantiles_pp4 %>% filter(Var1=="CHN", Var2=="USA") %>% pull(pp_prime4)

# quantiles_pp4 %>% arrange(desc(pp_prime4)) %>% print(n=100)
war_probs_pp4 <- quantiles_pp4 %>% filter(Var1=="CHN") %>% ggplot(aes(x=Var2, y=1-pp_q500, color="black")) +
  geom_point(size=.5) + 
  geom_point(aes(x=Var2, y=1-pp_prime4, color=bcOrange), size=.5) +
  geom_segment(aes(x=Var2, xend=Var2, y=1-pp_prime4, yend=1-pp_q500), color=bcOrange) +
  scale_color_manual("Equilibrium", values=c(bcOrange, "black"), labels=c("Preference Liberalization", "Baseline"), guide="legend") +
  geom_hline(yintercept=0, lty=2) +
  theme_classic() +
  theme(aspect.ratio=1) +
  labs(x="Directed Dyad", y="Probability of War", title="China: Probability of Invasion", subtitle="Change after preference liberalization")

### OLD ### 

# quantiles_pp %>% filter(Var2=="USA")

# point plot

# min_val <- min(quantiles_pp$pp_q975, na.rm=T)
# max_val <- max(quantiles_pp$pp_q025, na.rm=T)
# 
# quantiles_pp %>% filter(Var1!=Var2) %>% ggplot(aes(x=pp_q500, y=Var2)) + 
#   geom_point() +
#   geom_segment(aes(y=Var2, yend=Var2, x=pp_q025, xend=pp_q975)) +
#   scale_x_continuous(limits=c(min_val, max_val)) +
#   theme_classic() +
#   theme(axis.ticks.x=element_blank(),
#         axis.text.x=element_blank()) +
#  facet_wrap(~Var1, nrow=2)
