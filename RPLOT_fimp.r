a = read.table("FI_no_hsc")
b = read.table("FI_hsc")
c = read.table("FI_beta")

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
