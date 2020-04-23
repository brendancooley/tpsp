bcOrange <- "#BD6121"
hmColors <- colorRampPalette(c("white", bcOrange))(30)
naColor <- "#D3D3D3"

hm <- function(meltedDF, min_val, max_val, plot_title="", x_lab="", y_lab="") {
  out <- ggplot(data=meltedDF, aes(x=Var2, y=Var1, fill=value)) + 
    geom_tile(colour="white", width=.9, height=.9) +
    scale_fill_gradient(low=hmColors[1], high=hmColors[length(hmColors)], 
			breaks=c(min_val, max_val), limits=c(min_val, max_val), 				na.value=naColor) +
    theme_classic() +
    coord_fixed()  +
    labs(title=plot_title, x=x_lab, y=y_lab) +
    theme(legend.position = "none",
          axis.line=element_blank(),
          axis.text.x=element_text(angle=60, hjust=1),
          axis.ticks.x=element_blank(),
          axis.ticks.y=element_blank())
  return(out)
}