theme_js_bw_tickout <- function(..., bg='white'){
  require(grid)
  theme_bw(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(c(0.5,0.5,0.5,0.5), 'cm'),
          panel.background=element_rect(fill='transparent', color='black'),
          panel.border=element_rect(fill='transparent', color='black', size = 1),
          panel.grid=element_blank(),
          axis.title.x = element_text(color='black'), # vjust=0.1
          axis.title.y = element_text(color='black'), # vjust=0.1
          axis.ticks.length = unit(-0.2,"lines"),
          axis.ticks = element_line(color='black',size=0.4),
          axis.ticks.margin = unit(0.4,"lines"),
          axis.line = element_line(colour = 'black', size = 1),
          legend.title=element_blank(),
          legend.key=element_rect(fill='transparent', color='transparent'), strip.background = element_rect(fill = "grey85", 
                                                                                        colour = "grey20"))+theme(axis.text=element_text(colour="black",size=9))+
    theme(axis.text=element_text(colour="black",size=13))+
    theme(legend.text=element_text(color="black",size=13))+
    theme(axis.title = element_text(size=14))
}

theme_js_bw <- function(..., bg='white'){
  require(grid)
  theme_bw(...) +
    theme(rect=element_rect(fill=bg),
          plot.margin=unit(rep(0.5,4), 'lines'),
          panel.background=element_rect(fill='transparent', color='black'),
          panel.border=element_rect(fill='transparent', color='black', size = 1),
          panel.grid=element_blank(),
          axis.title = element_text(color='black', vjust=0.1),
          axis.ticks.length = unit(0.25,"lines"),
          axis.ticks = element_line(color='black',size=0.28),
          axis.ticks.margin = unit(0.25,"lines"),
          axis.line = element_line(colour = 'black'),
          legend.title=element_blank(),
          legend.key=element_rect(fill='transparent', color='transparent'), strip.background = element_rect(fill = "grey85", 
                                                                                                            colour = "grey20"))+theme(axis.text=element_text(colour="black",size=9))+
    theme(axis.text=element_text(colour="black",size=12))+
    theme(legend.text=element_text(color="black",size=12))+
    theme(axis.title = element_text(size=13))+theme(panel.grid=element_blank())
}

theme_js_classic_tickout <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +theme(axis.text=element_text(colour="black",size=9))+
    theme(axis.text=element_text(colour="black",size=12))+
    theme(legend.text=element_text(color="black",size=12))+
    theme(axis.title = element_text(size=13))+
    theme( axis.ticks.length = unit(-0.2,"lines"),
           axis.ticks = element_line(color='black',size=0.28),
           axis.ticks.margin = unit(0.4,"lines"),
           axis.line = element_line(colour = 'black', size = 1.25))
  
  
}

theme_js_classic <- function(..., bg='white'){
  require(grid)
  theme_classic(...) +theme(axis.text=element_text(colour="black",size=14))+
    theme(axis.text=element_text(colour="black",size=14))+
    theme(legend.text=element_text(color="black",size=14))+
    theme(axis.title = element_text(size=14))+
    theme( axis.ticks.length = unit(0.30,"lines"),
           axis.ticks = element_line(color='black',size=0.75),
           axis.ticks.margin = unit(0.4,"lines"),
           axis.line = element_line(colour = 'black', size = 1))
}

