
RES = list()

library(ggplot2)
library(reshape)

NN = c("RF","HS","BETA", "BETA-")
RES[[1]] = read.table("1")
rownames(RES[[1]]) = NN
RES[[1]] = melt(t(RES[[1]]))
RES[[2]] = read.table("2")
rownames(RES[[2]]) = NN
RES[[2]] = melt(t(RES[[2]]))
RES[[3]] = read.table("5")
rownames(RES[[3]]) = NN
RES[[3]] = melt(t(RES[[3]]))
RES[[4]] = read.table("10")
rownames(RES[[4]]) = NN
RES[[4]] = melt(t(RES[[4]]))
RES[[5]] = read.table("50")
rownames(RES[[5]]) = NN
RES[[5]] = melt(t(RES[[5]]))
RES[[6]] = read.table("100")
rownames(RES[[6]]) = NN
RES[[6]] = melt(t(RES[[6]]))

ntrees = c("1","2","5","10","50","100")
#ntrees = c("1","2","5","10","50")

names(RES) = ntrees 

RES_ALL = melt(RES)
RES_ALL$L1 = factor(RES_ALL$L1, levels=ntrees)
p1 <- ggplot(RES_ALL, aes(x = L1, y = value, fill=X2)) + 
  geom_boxplot(notch = TRUE, outlier.shape = NA) +
  #geom_violin(trim=FALSE) +
  #geom_line(data = EstimateMelted, aes(x = id,y = value, group = Type, colour = Type, linetype = Type), size = 1) +
  #scale_fill_manual(values=c("orange","cadetblue"))+ 
          #facet_grid(~group) + 
  #scale_color_manual(values=c("#000000","#CC6666")) +
  xlab("Number of trees") + ylab("Balanced accuracy")  + 
  #theme(axis.text.x = element_text(angle = 90)) +
  ylim(c(0.5,1))+
  theme(legend.position="bottom", legend.title =element_blank())+
  ggtitle("") +
  theme(plot.title = element_text(size=10))

#############################################
#### other stuff --> weighted vs non-weighted
a = read.table("PERF")
b = read.table("PERF_w")

RES = cbind(a,b)
colnames(RES) = c("non-weighted", "weighted")

library(reshape)
library(ggplot2)

RES_melt = melt(RES)
colnames(RES_melt) = c("Method","value")

p1 <- ggplot(RES_melt, aes(x = Method, y = value, fill=Method)) + 
  geom_boxplot(notch = FALSE, outlier.shape = NA) +
  #geom_violin(trim=FALSE) +
  #geom_line(data = EstimateMelted, aes(x = id,y = value, group = Type, colour = Type, linetype = Type), size = 1) +
  #scale_fill_manual(values=c("orange","cadetblue"))+ 
          #facet_grid(~group) + 
  #scale_color_manual(values=c("#000000","#CC6666")) +
  xlab("") + ylab("Balanced accuracy")  + 
  #theme(axis.text.x = element_text(angle = 90)) +
  ylim(c(0.7,0.9))+
  theme(legend.position="bottom", legend.title =element_blank())+
  ggtitle("") +
  theme(plot.title = element_text(size=15))
