#

NN = c(rep("VANILLA",20), rep("HS",20), rep("BETA",20))

d1 = as.matrix(read.table("0.0"))
rownames(d1) = NN
d2 = as.matrix(read.table("0.05"))
rownames(d2) = NN
d3 = as.matrix(read.table("0.1"))
rownames(d3) = NN
d4 = as.matrix(read.table("0.15"))
rownames(d4) = NN
d5 = as.matrix(read.table("0.2"))
rownames(d5) = NN

#RES = list(d1,d2,d3,d4,d5)
RES = list(d1,d3,d4,d5)

#names(RES) = c("0.0","0.05","0.1","0.15","0.2")
names(RES) = c("0.0","0.1","0.15","0.2")

#rel = as.factor(c("0.0","0.05","0.1","0.15","0.2"))
rel = as.factor(c("0.0","0.1","0.15","0.2"))


library(reshape)
library(ggplot2)


RES_ALL = melt(RES)
RES_ALL$L1 = factor(RES_ALL$L1, levels=rel)
p1 <- ggplot(RES_ALL, aes(x = X2, y = value, fill=X1)) + 
  geom_boxplot(outlier.shape = NA) + geom_boxplot(aes(color=X1)) +
  #geom_violin(trim=FALSE) +
  #geom_line(data = EstimateMelted, aes(x = id,y = value, group = Type, colour = Type, linetype = Type), size = 1) +
  #scale_fill_manual(values=c("orange","cadetblue"))+ 
          #facet_grid(~group) + 
  #scale_color_manual(values=c("#000000","#CC6666")) +
  xlab("") + ylab("Feature Importance")  + 
  #theme(axis.text.x = element_text(angle = 90)) +
  ylim(c(0,0.5))+
  theme_bw(base_size = 13) +
  theme(legend.position="bottom", legend.title =element_blank())+
  ggtitle("") +
  theme(plot.title = element_text(size=10)) +
  facet_grid("L1")#rows = vars(drv))

