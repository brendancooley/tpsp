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
milex_2030 <- read_csv(setup$M2030_raw_path, col_names=FALSE)

ccodes <- read_csv(setup$ccodes_path, col_names=FALSE) %>% pull(.)
N <- length(ccodes)
year <- read_csv(paste0(setup$data_path, "year.csv"), col_names=FALSE) %>% pull(.)

M <- bind_cols(ccodes, milex) %>% as_tibble()
colnames(M) <- c("ccode", "m")
M$m <- M$m / 1000000000
M$year <- "2011"

M2030 <- bind_cols(ccodes, milex_2030) %>% as_tibble()
colnames(M2030) <- c("ccode", "m")
# M2030$m <- M2030$m / 1000000000
M2030$year <- "2030"

# reorder by milex
M <- M %>% arrange(desc(m)) %>%
  mutate(ccode = factor(ccode, unique(ccode)))

base_fill <- "#808080"

milexFig <- ggplot(data=M, aes(x=ccode, y=m)) + 
  geom_bar(stat="identity", fill=base_fill, color="white") +
  theme_classic() +
  labs(title=paste0("Military Expenditure, ", year), x="", y="Military Expenditure (in billion $)") +
  theme(axis.text.x=element_text(angle=60, hjust=1))

ggsave(setup$f_milex_path)

Mprime <- bind_rows(M, M2030)
Mprime <- Mprime %>% arrange(desc(m)) %>%
  mutate(ccode = factor(ccode, unique(ccode)))

milex2030Fig <- Mprime %>% filter(ccode!="RoW") %>% ggplot(data=, aes(x=ccode, y=m, fill=year)) + 
  geom_bar(width=.75, position="dodge", stat="identity", color="white") +
  scale_fill_manual("Scenario", values=c(base_fill, bcOrange), labels=c("2011", "2030 Projection"), guide="legend") +
  theme_classic() +
  labs(title="Military Expenditure, 2011 and 2030 Projection", x="", y="Military Expenditure (in billion $)") +
  theme(axis.text.x=element_text(angle=60, hjust=1),
        aspect.ratio=1)
