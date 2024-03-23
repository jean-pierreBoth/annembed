### my macbook pro icloud path
setwd("/Users/jianshuzhao/Library/Mobile\ Documents/com~apple~CloudDocs/PICO_NEW")
library(ggplot2)
library(ggpp)
library(RColorBrewer)
### C elegan annembed
embed_elegan <- read.table("C_elegan_embedded_try.csv",sep=",", head=T)
head(embed_elegan)
library(ggplot2)

source("~/theme_js.R")
good.shapes = c(1:25,33:127)
embed_elegan$embryo.time.new = as.factor(embed_elegan$embryo.time.new)

df2 <- embed_elegan %>% group_by(cell.type) %>% summarize(annembed_1 = median(annembed_1), annembed_2 = median(annembed_2))

a = ggplot(data=embed_elegan,aes(x=annembed_1,y=annembed_2, shape=cell.type))+geom_point(aes(color=embryo.time.new,shape=cell.type),size=0.5, alpha=0.5) + scale_color_brewer(palette="Paired") + scale_shape_manual(values=good.shapes[1:37])

a +  theme_js_bw() + xlab("annembed_1") + ylab("annembed_2") + theme(legend.position="none") + ##+ geom_label(aes(label = cell.type)) ##+ guides(color = guide_legend(override.aes = list(size = 5))) + guides(shape = guide_legend(override.aes = list(size = 5)))## + theme(legend.position="none") 
  geom_dl(aes(label = cell.type), method = "smart.grid")

### median/mean to label each category
a + geom_text_repel(data = df2, aes(label = cell.type), size = 3.5, min.segment.length = 0, nudge_x = 1, max.overlaps=Inf) + theme_js_bw() + xlab("annembed_1") + ylab("annembed_2") + theme(legend.position="none")

## ggpp package using centroid

a + stat_centroid(aes(label =  cell.type), geom = "text_repel", position = position_nudge_keep(x = 1), 
                  size = 4, min.segment.length = 0, max.overlaps=Inf) +
  theme_js_bw() + xlab("annembed_1") + ylab("annembed_2") + theme(legend.position="none")
  ##guides(color = guide_legend(override.aes = list(size = 5))) + guides(shape = guide_legend(override.aes = list(size = 5)))

## use the voronoi algorithm to determine
a + geom_text_repel(aes(label = ifelse(is.voronoi_max(annembed_1, annembed_2, cell.type), cell.type, "")), max.overlaps=Inf) + theme_js_bw() + xlab("annembed_1") + ylab("annembed_2") + theme(legend.position="none")



### C elegan UWOT (Annoy)
library(uwot)
umap_elegan <- read.table("c-elegans_qc_final_all_sample_name.csv",sep=",", head=T, row.names=1) 
dim(umap_elegan)
head(umap_elegan)
umap <- umap(umap_elegan, n_neighbors = 15, min_dist = 0.001, verbose = TRUE, n_threads =8)
head(umap)
umap_matrix <- as.matrix(umap)
df <- data.frame(umap_matrix, embed_elegan$cell.type,embed_elegan$embryo.time.new)
head(df)
df3 <- df %>% group_by(embed_elegan.cell.type) %>% summarize(X1 = median(X1), X2 = median(X2))


b = ggplot(data=df,aes(x=X1,y=X2, shape=embed_elegan.cell.type))+geom_point(aes(color=embed_elegan.embryo.time.new,shape=embed_elegan.cell.type),size=0.5, alpha=0.5) + scale_color_brewer(palette="Paired") + scale_shape_manual(values=good.shapes[1:37])
b +  theme_js_bw() + xlab("UMAP_1") + ylab("UMAP_2") + theme(legend.position="none") + ##+ geom_label(aes(label = embed_elegan.cell.type))##+ guides(color = guide_legend(override.aes = list(size = 5))) + guides(shape = guide_legend(override.aes = list(size = 5)))## + theme(legend.position="none") 
  geom_dl(aes(label = embed_elegan.cell.type), method = "smart.grid")

##use ggrepel
b + geom_text_repel(data = df3, aes(label = embed_elegan.cell.type), size = 3.5, min.segment.length = 0, nudge_x = 1, max.overlaps=Inf) + theme_js_bw() + xlab("UMAP_1") + ylab("UMAP_2") + theme(legend.position="none")

### use ggpp
b + stat_centroid(aes(label =  embed_elegan.cell.type), geom = "text_repel", position = position_nudge_keep(x = 1), 
                  size = 4, min.segment.length = 0, max.overlaps=Inf) +
  theme_js_bw() + xlab("UMAP_1") + ylab("UMAP_2") + theme(legend.position="none")




a + geom_text_repel(data = df2, aes(label = cell.type), size = 3, min.segment.length = 0, nudge_x = 1) + theme_js_bw() + xlab("annembed_1") + ylab("annembed_2") + theme(legend.position="none")


b + geom_text_repel(aes(label = ifelse(is.voronoi_max(X1, X2, embed_elegan.cell.type), embed_elegan.cell.type, "")), max.overlaps=Inf) + theme_js_bw() + xlab("UMAP_1") + ylab("UMAP_2") + theme(legend.position="none")


### test add labels to each group

library(ggplot2)
library(directlabels)

ggplot(iris, aes(Petal.Length, Sepal.Length)) +
  geom_point(aes(color = Species)) +
  geom_dl(aes(color = Species, label = Species), method = "smart.grid")

#### another example

library(ggplot2)
library(ggrepel)

set.seed(42)

is.voronoi_max <- function(x, y, group) {
  x <- scales::rescale(x, c(0, 1))
  y <- scales::rescale(y, c(0, 1))
  rw <-  c(range(x), range(y))
  del <- deldir::deldir(x, y, z = group, rw =  rw, 
                        suppressMsge = TRUE)
  del_summ <- del$summary
  del_summ$index <- seq_len(nrow(del_summ))
  
  index <- vapply(split(del_summ, del_summ$z), 
                  function(x) x$index[which.max(x$dir.area)], 1)
  seq_len(length(x)) %in% index
}

ggplot(mtcars, aes(wt, mpg, color = factor(gear))) +
  geom_point() +
  geom_text_repel(aes(label = ifelse(is.voronoi_max(wt, mpg, gear), gear, "")))


### centroid example

library(ggplot2)
library(magrittr)
library(dplyr)

df1 <- mtcars
df2 <- mtcars %>% group_by(gear) %>% summarize(mpg = median(mpg), wt = median(wt))

ggplot(df1) +
  aes(wt, mpg, color = factor(gear)) +
  geom_point(size = 2) +
  geom_text_repel(data = df2, aes(label = gear), size = 8, min.segment.length = 0, nudge_x = 1) +
  theme_gray(base_size = 20)
