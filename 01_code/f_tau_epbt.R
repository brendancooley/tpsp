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
tau <- read_csv(setup$tau_path, col_names=FALSE)

ccodes <- read_csv(setup$ccodes_path, col_names=FALSE) %>% pull(.)
N <- length(ccodes)
year <- read_csv(paste0(setup$data_path, "year.csv"), col_names=FALSE) %>% pull(.)

hmColors <- colorRampPalette(c("white", bcOrange))(20)
naColor <- "#D3D3D3"

rectTrsp <- 0
rectColor <- "#000000CC"
rectStroke <- 1

tau_melted <- expand.grid(ccodes, ccodes)
tau_melted$tau <- tau %>% as.matrix() %>% as.vector()
tau_melted <- tau_melted %>% as_tibble()
colnames(tau_melted) <- c("Var1", "Var2", "value")

tau_mean <- tau_melted %>% filter(Var1 != Var2) %>% group_by(Var1) %>% summarise(tau_mean=mean(value))

min_val <- min(tau_melted$value, na.rm=T)
max_val <- max(tau_melted$value, na.rm=T)

tau_hm <- hm(tau_melted, min_val, max_val, x="Exporter", y="Importer", plot_title="Point Estimates: Policy Barriers to Trade")

ggsave(setup$f_tau_epbt_path)