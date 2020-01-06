# helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
# helperFiles <- list.files(helperPath)
# for (i in helperFiles) {
#   source(paste0(helperPath, i))
# }
# 
# libs <- c("tidyverse")
# ipak(libs)
# 
# analysisPath <- "../working/analysis/"
# dataPath <- paste0(analysisPath, "tpsp_data/")
# 
# milex <- read_csv(paste0(dataPath, "milex.csv"), col_names=FALSE)
# ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)
# Y <- read_csv(paste0(dataPath, "year.csv"), col_names=FALSE) %>% pull(.)

M <- bind_cols(ccodes, milex) %>% as_tibble()
colnames(M) <- c("ccode", "m")
M$m <- M$m / 1000000000

# reorder by milex
M <- M %>% arrange(desc(m)) %>%
  mutate(ccode = factor(ccode, unique(ccode)))

milexFig <- ggplot(data=M, aes(x=ccode, y=m)) + 
  geom_bar(stat="identity", fill=bcOrange) +
  theme_classic() +
  labs(title=paste0("Military Expenditure, ", Y), x="", y="Military Expenditure (in billion $)") +
  theme(axis.text.x=element_text(angle=60, hjust=1))
