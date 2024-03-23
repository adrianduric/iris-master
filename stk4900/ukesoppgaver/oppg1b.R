# Exercise 1b: Bootstrapping

# In this exercise we will explain bootstrapping in some more detail than the in Lecture 1.
# We will use the same data set as in Exercise 1, age of mineral samples.

age=c(249, 254, 243, 268, 253, 269, 287, 241, 273, 306, 303, 280, 260, 256, 278, 344, 304, 283, 310)

# First generate one bootstrap sample by the command

bootsamp=sample(age,replace=T)

# Compare this sampled data with the original data by writing
sort(bootsamp); sort(age)

# Compute the mean and median of the bootstrap sample and compare with the corresponding
#values in the original data.
mean(age); sd(age)
mean(bootsamp); sd(bootsamp)

# Draw another bootstrap sample and repeat the comparison.
bootsamp=sample(age,replace=T)
sort(bootsamp); sort(age)
mean(age); sd(age)
mean(bootsamp); sd(bootsamp)

# One will typically draw a large number of bootstrap samples, say 1000, calculate the
#statistic for which we want to find a confidence interval and use the 2.5 and 97.5 percentiles
#as the confidence limits. This can be done, for the mean, using the commands
bootagemean<-numeric(0)
for (i in 1:1000) bootagemean[i]<-mean(sample(age,replace=T))
sort(bootagemean)[c(25,975)]

# Compare with the results using the t-interval. Also, inspect the ten first bootstrap estimates
#of the mean by writing
bootagemean[1:10]

# Increase the number of bootstrap samples to 10000 and 50000. Comment.
# Make bootstrap confidence intervals for the median.
bootagemean<-numeric(0)
for (i in 1:10000) bootagemean[i]<-mean(sample(age,replace=T))
sort(bootagemean)[c(25,975)]

bootagemean<-numeric(0)
for (i in 1:50000) bootagemean[i]<-mean(sample(age,replace=T))
sort(bootagemean)[c(25,975)]

# Vi ser at konfidensintervallet krymper for flere bootstrap-estimater, som gir mening siden standardavviket synker med antall samples, og vi finner at gj.snittet til det opprinnelige datasettet ligger utenfor 95% KI.

bootagemedian<-numeric(0)
for (i in 1:50000) bootagemedian[i]<-median(sample(age,replace=T))
sort(bootagemedian)[c(25,975)]