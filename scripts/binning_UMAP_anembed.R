### mmgenome2
if(!require(remotes))
  install.packages("remotes")
remotes::install_github("jianshu93/mmgenome2")
library(mmgenome2)

setwd("/Users/jianshuzhao")
library(mmgenome2)
library(Rtsne.multicore)
library(Rtsne)
 
mm <- mmload(
  assembly = "OMZ_S1220_bin_check/S138_final.contigs.new.fa",
  list(
    nameofcoverage1 = read.csv("/Users/jianshuzhao/OMZ_S1220_bin_check/S138.contig_abundance.metabat.csv"),
    nameofcoverage2 = read.csv("/Users/jianshuzhao/OMZ_S1220_bin_check/S138.contig_abundance.metabat.csv")
  ),
  taxonomy = "OMZ_S1220_bin_check/taxonomy_new.csv",
  verbose = TRUE,
  kmer_pca = TRUE,
  kmer_BH_tSNE = TRUE,
  num_threads = 8
)

head(mm$tSNE1)
mmstats(mm)
mm$cov_nameofcoverage1


a=mmplot(mm,
       x = "tSNE1",
       y = "cov_nameofcoverage1",
       y_scale = "log10",
       color_by = "phylum",
       color_vector=c("#AECDE1", "#CC7AA6",
                      "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                      "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                      "#C7BAA6", "#F7CCAD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                      "#AECDE1", "#CC7AA6",
                      "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                      "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                      "#C7BAA6", "#F7CCAD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                      "#AECDE1", "#CC7AA6",
                      "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                      "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                      "#C7BAA6", "#F7CCAD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                      "#AECDE1", "#CC7AA6",
                      "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                      "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                      "#C7BAA6", "#F7CCAD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                      "#AECDE1", "#CC7AA6",
                      "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                      "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                      "#C7BAA6", "#F7CCAD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                      "#AECDE1", "#CC7AA6",
                      "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                      "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                      "#C7BAA6", "#F7CCAD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                      "#AECDE1", "#CC7AA6",
                      "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                      "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                      "#C7BAA6", "#F7CCAD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                      "#AECDE1", "#CC7AA6",
                      "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                      "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                      "#C7BAA6", "#F7CCAD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#599861","#666666",
                      "#FCFAF0","#AD6F3B","black",
                      "#5F7FC7","#F5B369","#DA5724",
                      "#508578","#CD9BCD","#8569D5", "#D14285","#599861"))
source("theme_js.R")
colors_new <- c("#AECDE1",
           "#644194", "#8569D5","#F5E369","#999999",
           "#652926","#8569D5", "#D14285",
           "#C7BAA6", "#F7CCAD","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
           "#AECDE1", "#CC7AA6",
           "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
           "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
           "#C7BAA6", "#F7CCAD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
           "#AECDE1", "#CC7AA6",
           "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
           "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
           "#C7BAA6", "#F7CCAD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
           "#AECDE1", "#CC7AA6",
           "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
           "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
           "#C7BAA6", "#F7CCAD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
           "#AECDE1", "#CC7AA6",
           "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
           "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
           "#C7BAA6", "#F7CCAD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
           "#AECDE1", "#CC7AA6",
           "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
           "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
           "#C7BAA6", "#F7CCAD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
           "#AECDE1", "#CC7AA6",
           "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
           "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
           "#C7BAA6", "#F7CCAD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
           "#AECDE1", "#CC7AA6",
           "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
           "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
           "#C7BAA6", "#F7CCAD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#599861","#666666",
           "#FCFAF0","#AD6F3B","black",
           "#5F7FC7","#F5B369","#DA5724",
           "#508578","#CD9BCD","#8569D5", "#D14285","#599861")
a+theme_js_classic()+scale_color_manual(values=colors_new)

ggsave("save.pdf", width = 20, height = 20)



mm <- mmload(
  assembly = "min17_DAS/contigs.fasta",
  list(
    nameofcoverage1 = read.csv("/Users/jianshuzhao/min17_DAS/contig_abundance.metabat_new.csv"),
    nameofcoverage2 = read.csv("/Users/jianshuzhao/min17_DAS/contig_abundance.metabat_new.csv")
  ),
  taxonomy = "min17_DAS/DAS_bin_contig.csv",
  verbose = TRUE,
  kmer_pca = TRUE,
  kmer_BH_tSNE = TRUE,
  num_threads = 8
)

head(mm$tSNE1)
mmstats(mm)
mm$cov_nameofcoverage1

a=mmplot(mm,
         x = "tSNE1",
         y = "cov_nameofcoverage1",
         y_scale = "log10",
         color_by = "phylum")


a+theme_js_classic()+scale_color_manual(values=colors_new)

ggsave("save.pdf", width = 20, height = 20)


a=mmplot(mm,
         x = "PC1",
         y = "cov_nameofcoverage1",
         y_scale = "log10",
         color_by = "phylum")


a+theme_js_classic()+scale_color_manual(values=colors_new)






b=mmplot(mm,
         x = "PC1",
         y = "cov_nameofcoverage1",
         y_scale = "log10",
         color_by = "phylum",
         color_vector=c("#AECDE1", "#CC7AA6",
                        "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                        "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                        "#C7BAA6", "#F7CCAD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                        "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                        "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                        "#C7BAA6", "#F7CCAD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                        "#AECDE1", "#CC7AA6",
                        "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                        "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                        "#C7BAA6", "#F7CCAD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#8569D5", "#D14285","#599861",
                        "#AECDE1", "#CC7AA6",
                        "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                        "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                        "#C7BAA6", "#F7CCAD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#8569D5", "#D14285","#599861"))
b+theme_js_classic()+scale_color_manual(values=colors_new)


c=mmplot(mm,
         x = "gc",
         y = "cov_nameofcoverage1",
         y_scale = "log10",
         color_by = "phylum",
         color_vector=c("#AECDE1", "#CC7AA6",
                        "#644194", "#8569D5","#F5E369","#999999", "#B3DE8C",
                        "#652926", "#C2CCD6","#8569D5", "#D14285","#4AF2A1",
                        "#C7BAA6", "#F7CCAD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#599861","#666666",
                        "#FCFAF0","#AD6F3B","black",
                        "#5F7FC7","#F5B369","#DA5724",
                        "#508578","#CD9BCD","#8569D5", "#D14285","#599861"))
c+theme_js_classic()

