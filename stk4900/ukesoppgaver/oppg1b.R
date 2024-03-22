# Exercise 1b: Bootstrapping
# In this exercise we will explain bootstrapping in some more detail than the in Lecture 1.
# We will use the same data set as in Exercise 1, age of mineral samples.
# First generate one bootstrap sample by the command
bootsamp=sample(rock.age,replace=T)
# Compare this sampled data with the original data by writing
sort(bootsamp); sort(rock.age)
# Compute the mean and median of the bootstrap sample and compare with the corresponding
#values in the original data.
# Draw another bootstrap sample and repeat the comparison.
# One will typically draw a large number of bootstrap samples, say 1000, calculate the
#statistic for which we want to find a confidence interval and use the 2.5 and 97.5 percentiles
#as the confidence limits. This can be done, for the mean, using the commands
bootagemean<-numeric(0)
for (i in 1:1000) bootagemean[i]<-mean(sample(rock.age,replace=T))
sort(bootagemean)[c(25,975)]
# Compare with the results using the t-interval. Also, inspect the ten first bootstrap estimates
#of the mean by writing
bootagemean[1:10]
# Increase the number of bootstrap samples to 10000 and 50000. Comment.
# Make bootstrap confidence intervals for the median.