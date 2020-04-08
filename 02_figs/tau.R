helperPath <- "~/Dropbox (Princeton)/14_Software/R/"
helperFiles <- list.files(helperPath)
for (i in helperFiles) {
  source(paste0(helperPath, i))
}

libs <- c("tidyverse")
ipak(libs)

tpspPath <- "~/Dropbox (Princeton)/1_Papers/tpsp/01_data/"
dataPath <- paste0(tpspPath, "data/", size)
resultsPath <- paste0(tpspPath, "results/", size)
estimatesPath <- paste0(resultsPath, "estimates/")

ccodes <- read_csv(paste0(dataPath, "ccodes.csv"), col_names=FALSE)
Y <- read_csv(paste0(dataPath, "year.csv"), col_names=FALSE) %>% pull(.)
tau <- read_csv(paste0(dataPath, "tau.csv"), col_names=FALSE)

hmColors <- colorRampPalette(c("white", bcOrange))(20)
naColor <- "#D3D3D3"

rectTrsp <- 0
rectColor <- "#000000CC"
rectStroke <- 1

tauhm <- function(tau, highlight=NULL) {
  colnames(tau) <- ccodes %>% pull(.)
  tau <- bind_cols(tau, ccodes) 
  colnames(tau)[colnames(tau)=="X1"] <- "j_iso3"
  tauDF <- tau %>% gather("i_iso3", "tau_ji", -j_iso3)
  tauDF$tau_ji <- ifelse(tauDF$tau_ji==0, NA, tauDF$tau_ji)
  tauDF$tau_ji <- as.numeric(tauDF$tau_ji)

  out <- ggplot(tauDF, aes(x=i_iso3, y=j_iso3, fill=tau_ji)) +
    geom_tile(colour="white", width=.9, height=.9) +
    scale_fill_gradient(low=hmColors[1], high=hmColors[length(hmColors)], 
                        breaks=c(min(tauDF$tau_ji, na.rm=TRUE), max(tauDF$tau_ji, na.rm=TRUE)), labels=c("Low", "High"), 
                        guide="colorbar", na.value=naColor) +
    labs(x='Exporter', y='Importer', title=paste0("Policy Barriers to Trade, ", Y)) +
    labs(fill="Policy Barrier") +
    theme_classic() +
    coord_fixed() +
    theme(axis.text.x=element_text(angle=60, hjust=1),
          axis.ticks.y=element_blank(),
          axis.ticks.x=element_blank(),
          axis.line.y=element_blank(),
          axis.line.x=element_blank())

  if (!is.null(highlight)) {
    cc <- which(ccodes==highlight)
    out <- out + annotate("rect", ymin=cc-.5, ymax=cc+.5, xmin=.5, xmax=nrow(ccodes)+.5, alpha=rectTrsp, color=rectColor, size=rectStroke)
  }
  
  return(out)
}

# tauhm(tau, highlight="USA")
