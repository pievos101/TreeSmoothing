## FINAL FIMP PLOTS

a1 = read.table("50weight_500samps/FI_no_hsc")
b1 = read.table("50weight_500samps/FI_hsc")
c1 = read.table("50weight_500samps/FI_beta")

a2 = read.table("80weight_500samps/FI_no_hsc")
b2 = read.table("80weight_500samps/FI_hsc")
c2 = read.table("80weight_500samps/FI_beta")

a3 = read.table("90weight_500samps/FI_no_hsc")
b3 = read.table("90weight_500samps/FI_hsc")
c3 = read.table("90weight_500samps/FI_beta")

a4 = read.table("95weight_500samps/FI_no_hsc")
b4 = read.table("95weight_500samps/FI_hsc")
c4 = read.table("95weight_500samps/FI_beta")


# Coverage 
A1_cov = apply(a1,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
B1_cov = apply(b1,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
C1_cov = apply(c1,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20

A2_cov = apply(a2,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
B2_cov = apply(b2,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
C2_cov = apply(c2,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20

A3_cov = apply(a3,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
B3_cov = apply(b3,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
C3_cov = apply(c3,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20

A4_cov = apply(a3,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
B4_cov = apply(b3,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
C4_cov = apply(c3,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20

RES1 = cbind(A1_cov, B1_cov, C1_cov)
RES2 = cbind(A2_cov, B2_cov, C2_cov)
RES3 = cbind(A3_cov, B3_cov, C3_cov)
RES4 = cbind(A4_cov, B4_cov, C4_cov)

MEAN1 = apply(RES1,2,mean)
MEAN2 = apply(RES2,2,mean)
MEAN3 = apply(RES3,2,mean)
MEAN4 = apply(RES4,2,mean)

STRR1 = apply(RES1,2,sd)/dim(RES1)[1]
STRR2 = apply(RES2,2,sd)/dim(RES2)[1]
STRR3 = apply(RES3,2,sd)/dim(RES3)[1]
STRR4 = apply(RES4,2,sd)/dim(RES4)[1]

MEAN = rbind(MEAN1,MEAN2,MEAN3,MEAN4)
STRR = rbind(STRR1, STRR2, STRR3, STRR3)
CN = c("RF","HS","BETA")
RN = c("50-50","80-20","90-10","95-5")
RN = 1:4
colnames(MEAN) = CN
colnames(STRR) = CN
rownames(MEAN) = RN
rownames(STRR) = RN

library(reshape)
library(ggplot2)

ALL = melt(MEAN)
ALL = cbind(ALL, as.vector(STRR))

colnames(ALL) = c("ratio","Method","mean","se")

p1 <- ggplot(ALL, aes(x=ratio, y=mean, colour=Method)) + 
    geom_errorbar(aes(ymin=mean-se, ymax=mean+se), width=.2) +
    geom_line(size=1) +
    geom_point(size=2) +
    ylim(0.8,1) +
    theme(legend.position="bottom", legend.title =element_blank())+
    ylab("Mean Coverage +- SE") +
    xlab("Class Ratio") +
    theme_bw(base_size = 13) + 
    scale_x_continuous(breaks = 1:4, 
    labels= c("50-50","80-20","90-10","95-5"))


colnames(RES1) = CN
colnames(RES2) = CN
colnames(RES3) = CN
colnames(RES4) = CN

BOX = list(RES1, RES2, RES3, RES4)
names(BOX) = c("50-50","80-20","90-10","95-5")

BOX_melted = melt(BOX)
colnames(BOX_melted) = c("X1", "Method", "value", "L1")

p2 <- ggplot(BOX_melted, aes(x=L1, y=value, fill=Method)) + 
    geom_boxplot()+
    ylim(0.5,1) +
    theme(legend.position="bottom", legend.title =element_blank())+
    ylab("Coverage") +
    xlab("Class Ratio") +
    theme_bw(base_size = 13) #+ 
    #scale_x_continuous(breaks = 1:4, 
    #labels= c("50-50","80-20","90-10","95-5"))




#########################################################
a = read.table("FI_no_hsc")
b = read.table("FI_hsc")
c = read.table("FI_beta")

# Coverage 
A_cov = apply(a,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
B_cov = apply(b,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20
C_cov = apply(c,1,function(x){sum(is.element(1:20,rank(-as.matrix(x))[1:20]))})/20

RES1 = cbind(A_cov, B_cov, C_cov)

# Quality 
A_qual = apply(a,1, function(x){x=x/sum(x);sqrt((mean(x[1:20]) - mean(x[-(1:20)]))^2)})
B_qual = apply(b,1, function(x){x=x/sum(x);sqrt((mean(x[1:20]) - mean(x[-(1:20)]))^2)})
C_qual = apply(c,1, function(x){x=x/sum(x);sqrt((mean(x[1:20]) - mean(x[-(1:20)]))^2)})

RES2 = cbind(A_qual, B_qual, C_qual)

# Mean Coverage Standard Error

MEAN = apply(RES1,2,mean)
STER = apply(RES1,2,sd)/dim(RES1)[1]

NN = c("RF","HS","BETA")
colnames(RES1) = NN
boxplot(RES1, ylim=c(0.2, 1), ylab="Coverage")
points(1:3,MEAN, pch=19, type="b", lwd=2, col="orange", cex=0.5)
arrows(1:3,MEAN-STER,1:3,MEAN+STER, code=3, length=0.2, angle = 90)

#colnames(RES2) = NN
#dev.off()
#par(mfrow=c(1,2))
#boxplot(RES1, ylim=c(0, 1))
#boxplot(RES2)


################################################
#### OTHER STUFF
##################################################
par(mfrow=c(1,3))
A = cbind(unlist(a[,1:20]), unlist(a[,-(1:20)]))
boxplot(A)
B = cbind(unlist(b[,1:20]), unlist(b[,-(1:20)]))
boxplot(B)
C = cbind(unlist(c[,1:20]), unlist(c[,-(1:20)]))
boxplot(C)

###################################################
par(mfrow=c(1,3))
barplot(t(as.matrix(a)))
barplot(t(as.matrix(b)))
barplot(t(as.matrix(c)))


a_cov = sum(is.element(1:20,rank(-as.matrix(a))[1:20]))
b_cov = sum(is.element(1:20,rank(-as.matrix(b))[1:20]))
c_cov = sum(is.element(1:20,rank(-as.matrix(c))[1:20]))


RES = c(a_cov, b_cov, c_cov)/20
print(RES)


a = as.matrix(a)
b = as.matrix(b)
c = as.matrix(c)

a = a/sum(a)
b = b/sum(b)
c = c/sum(c)

as = sqrt((mean(a[1:20]) - mean(a[-(1:20)]))^2)
bs = sqrt((mean(b[1:20]) - mean(b[-(1:20)]))^2)
cs = sqrt((mean(c[1:20]) - mean(c[-(1:20)]))^2)

print(as)
print(bs)
print(cs)
