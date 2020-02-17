# helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
# helperFiles <- list.files(helperPath)
# for (i in helperFiles) {
#   source(paste0(helperPath, i))
# }
# 
# libs <- c("tidyverse", "patchwork")
# ipak(libs)
# 
# tpspPath <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
# dataPath <- paste0(tpspPath, "tpsp_data_mini/")
# resultsPath <- paste0(tpspPath, "results_rcv_ft/")
# 
# milex <- read_csv(paste0(dataPath, "milex.csv"), col_names=FALSE)
# ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)
# W <- read_csv(paste0(dataPath, "cDists.csv"), col_names=FALSE)
# W <- log(W)
# N <- length(ccodes %>% pull(.))
# 
# gamma_tilde <- read_csv(paste0(resultsPath, "gamma_tilde.csv"), col_names=FALSE) %>% pull(.)
# alpha_tilde <- read_csv(paste0(resultsPath, "alpha_tilde.csv"), col_names=FALSE) %>% pull(.)
# 
# W <- read_csv(paste0("~/Dropbox (Princeton)/1_Papers/tpsp/01_data/data/mini/cDists.csv"), col_names=FALSE)
# milex <- read_csv(paste0("~/Dropbox (Princeton)/1_Papers/tpsp/01_data/data/mini/milex.csv"), col_names=FALSE)
# ccodes <- read_csv(paste0("~/Dropbox (Princeton)/1_Papers/tpsp/01_data/data/mini/ccodes.csv"), col_names=FALSE)
# N <- length(ccodes %>% pull(.))

chi_ji <- function(j, i) {
  m_j <- milex[j, ] %>% pull(.)
  m_i <- milex[i, ] %>% pull(.) 
  d_ji <- W[j, i] %>% pull(.)
  rho_ji <- exp(-alpha1*d_ji)
  out <-  m_j * rho_ji / (m_j * rho_ji + m_i)
  return(out)
}

d_ji <- W[4,5] %>% pull(.)
rho_ji <- exp(-alpha1*d_ji)

chi <- matrix(data=NA, nrow=N, ncol=N)

for (i in 1:nrow(ccodes)) {
  for (j in 1:nrow(ccodes)) {
    if (i != j) {
      if (ccodes[i, ] %>% pull(.) != "RoW" & ccodes[j, ] %>% pull(.) != "RoW") {
        chi[j, i] <- chi_ji(j, i)
      }
    }
  }
}

chi <- as_tibble(chi)
minchi <- 0
maxchi <- 1

hmColors <- colorRampPalette(c("white", bcOrange))(30)
naColor <- "#D3D3D3"

chihm <- function(chi, minchi, maxchi) {
  colnames(chi) <- ccodes %>% pull(.)
  chi <- bind_cols(chi, ccodes) 
  colnames(chi)[colnames(chi)=="X1"] <- "j_iso3"
  chiDF <- chi %>% gather("i_iso3", "chi_ji", -j_iso3)
  chiDF$chi_ji <- as.numeric(chiDF$chi_ji)
  chiDF <- chiDF %>% filter(i_iso3 != "RoW", j_iso3 != "RoW")
  # rcvDF %>% print(n=50)
  
  ggplot(chiDF, aes(x=i_iso3, y=j_iso3, fill=chi_ji)) +
    geom_tile(colour="white", width=.9, height=.9) +
    # geom_text(color="black", aes(label=round(chi_ji, 2))) +
    scale_fill_gradient(low=hmColors[1], high=hmColors[length(hmColors)], 
                        breaks=c(minchi, maxchi), limits=c(minchi, maxchi), labels=c("0", "1"), 
                        guide="colorbar", na.value=naColor) +
    labs(x='Defender', y='Attacker', title=paste0("Probability of Successful Conquest")) +
    labs(fill="Value") +
    theme_classic() +
    coord_fixed() +
    theme(axis.text.x=element_text(angle=60, hjust=1),
          axis.ticks.y=element_blank(),
          axis.ticks.x=element_blank(),
          axis.line.y=element_blank(),
          axis.line.x=element_blank(),
          legend.position="none")
}

# chihm(chi, minchi, maxchi)

