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

milex <- read_csv(setup$M_path, col_names=FALSE)

ccodes <- read_csv(setup$ccodes_path, col_names=FALSE) %>% pull(.)
N <- length(ccodes)
year <- read_csv(paste0(setup$data_path, "year.csv"), col_names=FALSE) %>% pull(.)

M <- bind_cols(ccodes, milex) %>% as_tibble()
colnames(M) <- c("ccode", "m")
M$m <- M$m / 1000000000

# reorder by milex
M <- M %>% arrange(desc(m)) %>%
  mutate(ccode = factor(ccode, unique(ccode)))

milexFig <- ggplot(data=M, aes(x=ccode, y=m)) + 
  geom_bar(stat="identity", fill=bcOrange) +
  theme_classic() +
  labs(title=paste0("Military Expenditure, ", year), x="", y="Military Expenditure (in billion $)") +
  theme(axis.text.x=element_text(angle=60, hjust=1))

ggsave(setup$f_milex_path)