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

NN = c("RF","HS","BETA")
colnames(RES1) = NN
colnames(RES2) = NN
#dev.off()
par(mfrow=c(1,2))
boxplot(RES1, ylim=c(0, 1))
boxplot(RES2)



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
