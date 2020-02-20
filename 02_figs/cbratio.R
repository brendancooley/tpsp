# helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
# helperFiles <- list.files(helperPath)
# for (i in helperFiles) {
#   source(paste0(helperPath, i))
# }
# 
# libs <- c("tidyverse", "patchwork", "ggpubr")
# ipak(libs)
# 
# estimatesPath <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/results/archive_fmir/estimates/"
# c_hat <- read_csv(paste0(estimatesPath, "c_hat.csv"), col_names=FALSE) %>% pull(.)
# rcv_eq <- read_csv(paste0(estimatesPath, "rcv_eq.csv"), col_names=FALSE)
# ccodes <- read_csv("~/Dropbox (Princeton)/1_Papers/tpsp/01_data/data/mini/ccodes.csv", col_names=FALSE)

rcv_eq2 <- rcv_eq

colnames(rcv_eq2) <- ccodes %>% pull(.)
rcv_eq2 <- cbind(ccodes, rcv_eq2) %>% as_tibble()
rcv_eq_long <- rcv_eq2 %>% pivot_longer(ccodes %>% pull(.), names_to="def", values_to="rcv")
colnames(rcv_eq_long)[1] <- "att"

rcv_eq_long <- rcv_eq_long %>% filter(att != def) %>% filter(att!="RoW") %>% filter(def!="RoW")
rcv_eq_long$cb_ratio <- rcv_eq_long$rcv / c_hat

rcv_eq_long$dyad <- paste0(rcv_eq_long$att, "-", rcv_eq_long$def)

rcv_eq_long <- rcv_eq_long %>% arrange(desc(cb_ratio))
rcv_eq_long$dyad <- factor(rcv_eq_long$dyad, levels = rcv_eq_long$dyad[order(rcv_eq_long$cb_ratio, decreasing=TRUE)])

cb_ratio_plot <- ggplot(data=rcv_eq_long, aes(x=dyad, y=cb_ratio)) + 
  geom_linerange(aes(x=dyad, ymin=0, ymax=cb_ratio), size=1.5, color="lightgray") +
  geom_point(aes(color=att), size=2.5) +
  color_palette("jco") +
  theme_classic() +
  labs(x="Attacker-Defender", y="Estimated Inverse Cost-Benefit Ratio", color="Attacker") +
  theme(axis.text.x=element_text(angle=45, hjust=1))
