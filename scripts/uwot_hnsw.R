library(uwot)
library(RcppHNSW)
library(devtools) 
##library(Rtsne) # Load package
library(ggplot2)
devtools::install_github("jlmelville/snedata")
mnist <- snedata::download_fashion_mnist()
###default annoy implenmentation
Sys.time()
mnist_umap <- umap(mnist[,1:784], n_neighbors = 100, min_dist = 0.001, verbose = TRUE, n_threads = 10)
Sys.time()

write.csv(file="mnist_fashion.csv",mnist[,1:784],row.names = FALSE)



### hnsw implenmentation
Sys.time()
nbrs <- hnsw_knn(as.matrix(mnist[,1:784]), k = 100, n_threads = 10, M=100, ef_construction = 200, ef=100)
Sys.time()
umap_hnsw <- umap(mnist[,1:784], min_dist = 0.001, verbose = TRUE, nn_method = nbrs)
Sys.time()



str(mnist_umap)
head(mnist_umap)
a <- as.matrix(mnist_umap)
write.csv(file="out.csv",a)
df <- data.frame(a,mnist$Label)
head(df)
f1=ggplot(data=df)+geom_point(aes(x=X1,y=X2,color=mnist.Label),size=0.1) +
  theme_classic() + theme(axis.text=element_text(colour="black",size=14))+
  theme(legend.text=element_text(color="black",size=13),
        legend.title=element_text(color="black",size=13))+
  theme(axis.title = element_text(size=14))
f1 + labs(x="UMAP 1",y="UMAP 2")
ggsave(filename="uwot.mnist_fashion.pdf",f1)

## annembed results, fast and efficient
## running cargo build --release --features intel-mkl in the crate folder
ups <- read.table(file="mnist_fashion.csv",sep=",")
ups$V1 <- as.factor(ups$V1)
head(ups)
f1=ggplot(data=ups)+geom_point(aes(x=V2,y=V3,color=V1),size=0.1) +
  theme_classic() + theme(axis.text=element_text(colour="black",size=14))+
  theme(legend.text=element_text(color="black",size=13),
        legend.title=element_text(color="black",size=13))+
  theme(axis.title = element_text(size=14))
f1 + labs(x="annembed 1",y="annembed 2")


library(reshape2)
### higgs dataset scaling
ups <- read.table(file="annembed_scaling_higgs.txt",sep="\t",row.names = 1, header = T)
head(ups)
ups$threads <- as.factor(ups$threads)
ups.melt<-melt(ups,id.var=c("threads"))
head(ups.melt)
ups.melt
f1 = ggplot(data=ups.melt,aes(x=threads,y=value, fill=variable))+geom_bar(stat="identity", color="black", position=position_dodge())+
  theme_classic() + theme(axis.text=element_text(colour="black",size=14))+
  theme(legend.text=element_text(color="black",size=13),
        legend.title=element_text(color="black",size=13))+
  theme(axis.title = element_text(size=14)) +
  scale_fill_manual(values=c("#508578", "#DA5724","#5F7FC7")) + scale_y_continuous(expand=c(0,0))
f1 + labs(x="Number of NUMA threads",y="Time(min)",fill="")

###PBMCs dataset includes a count matrix that measures 3000 genes in 3000 different cells.
PBMC <- read.table(file="counts_PBMC.csv",sep=",",row.names = 1, header = T)
dim(PBMC)
annotation <- read.table(file="celltypes_PBMC.txt",sep="\t")
dim(annotation)
Sys.time()
PBMC_umap <- umap(PBMC, n_neighbors = 15, min_dist = 0.01, verbose = TRUE, n_threads = 10)
Sys.time()
a <- as.matrix(PBMC_umap)
results <- data.frame(annotation$V1,a)
head(results)

f1=ggplot(data=results)+geom_point(aes(x=X1,y=X2,color=annotation.V1 ),size=0.5) +
  theme_classic() + theme(axis.text=element_text(colour="black",size=14))+
  theme(legend.text=element_text(color="black",size=13),
        legend.title=element_text(color="black",size=13))+
  theme(axis.title = element_text(size=14)) +
  scale_color_manual(values=c("#AECDE1", "#CC7AA6",
                              "#644194","#F5E369","#999999", "#B3DE8C",
                              "#652926", "#D14285","#4AF2A1",
                              "#C7BAA6", "#F7CCAD","#599861"))
f1 + labs(x="UMAP 1",y="UMAP 2",fill="Cell Type")


### PBMC dataset using annembed
PBMC_annembed <- read.table(file="embedded.csv",sep=",")
dim(PBMC)
annotation <- read.table(file="celltypes_PBMC.txt",sep="\t")

a <- as.matrix(PBMC_annembed)
results <- data.frame(annotation$V1,a)
head(results)
f1=ggplot(data=results)+geom_point(aes(x=V1,y=V2,color=annotation.V1 ),size=0.5) +
  theme_classic() + theme(axis.text=element_text(colour="black",size=14))+
  theme(legend.text=element_text(color="black",size=13),
        legend.title=element_text(color="black",size=13))+
  theme(axis.title = element_text(size=14)) +
  scale_color_manual(values=c("#AECDE1", "#CC7AA6",
                             "#644194","#F5E369","#999999", "#B3DE8C",
                             "#652926", "#D14285","#4AF2A1",
                             "#C7BAA6", "#F7CCAD","#599861"))
f1 + labs(x="Annembed 1",y="Annembed 2",fill="Cell Type")





## running annembed using command line mode


mnist_annembed <- read.table(file="embedded_mnist.csv",sep=",")
results <- data.frame(mnist_annembed,mnist$Label)
head(results)
f1=ggplot(data=results)+geom_point(aes(x=V1,y=V2,color=mnist.Label),size=0.5) +
  theme_classic() + theme(axis.text=element_text(colour="black",size=14))+
  theme(legend.text=element_text(color="black",size=13),
        legend.title=element_text(color="black",size=13))+
  theme(axis.title = element_text(size=14)) +
  scale_color_manual(values=c("#AECDE1", "#CC7AA6",
                              "#644194","#F5E369","#999999", "#B3DE8C",
                              "#652926", "#D14285","#4AF2A1",
                              "#C7BAA6", "#F7CCAD","#599861"))
f1 + labs(x="Annembed 1",y="Annembed 2",fill="Cell Type")



### ncbi refseq all genomes aa, setsketch

refseq_annembed <- read.table(file="Refseq_annembed_aa_setsketch.csv",sep=",")
head(refseq_annembed)
f1=ggplot(data=refseq_annembed)+geom_point(aes(x=V2,y=V3),size=0.1, color="black", alpha=0.2) +
  theme_classic() + theme(axis.text=element_text(colour="black",size=14))+
  theme(legend.text=element_text(color="black",size=13),
        legend.title=element_text(color="black",size=13))+
  theme(axis.title = element_text(size=14)) 
f1 + labs(x="Annembed 1",y="Annembed 2")
  