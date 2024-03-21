# Exercise 1:  one-sample t-test and confidence interval



# You may copy the commands below from the web-browser into the command-window of R (or into a R-script)

# Check this small introduction to R in case you need it.

# A line that starts with # is a comment, and R will disregard such lines.


#At the lectures we looked an example with the age of mineral samples (cf. slide 3 from the lectures)

#We will in this exercise see how the computations for this examples may be done in R.





#Start by reading the data into R. This may be done by the command:

age=c(249, 254, 243, 268, 253, 269, 287, 241, 273, 306, 303, 280, 260, 256, 278, 344, 304, 283, 310)





#Compute mean, median and standard deviation:

mean(age)

median(age)

sd(age)



# Check that you get the same result as in the lectures (cf slide 3)





# Make a histogram (cf. slide 4)

hist(age)





# Plot the empirical distribution function (cf. slide 4)

plot(ecdf(age))                                         # Basic plot

plot(ecdf(age),verticals=T, do.points=F)          # Nicer looking plot





# Compute min, first quartile, median, third quartile, and max (cf. slide 5)

quantile(age)





# Make a boxplot (cf. slide 5)

boxplot(age)



#We will then consider confidence intervals and hypothesis testing.

# It is important that you get familiar with the basic commands in R to:

# generate samples from a distribution (e.g. rnorm) ; estimate the value of the density function at a given value (e.g. dnorm) ;

# obtain the value of the cumulative density function (CDF) at a given point (e.g. pnorm); and get the value for a certain quantile (inverse of the CDF, e.g. qnorm)

# See for example: https://stat.ethz.ch/R-manual/R-devel/library/stats/html/TDist.html 



# Plot the density function and the CDF for a t-distribution with 18 degrees of freedom in the range  -5 to 5 (hint: use the "seq" command the generate the range)

x = seq(-5, 5, 0.01)
y_density = dt(x, df=18)
y_cumul = pt(x, df=18)

plot(x, y_density, type="l")
plot(x, y_cumul, verticals=T, do.points=F)

# Compute the 97.5% percentile of the t-distribution with 18 degrees of freedom:

qt(0.975,18)

# Add a vertical line to the density and CDF plot at the 0.975 quantile. Hint: use the command abline

plot(x, y_density, type="l")
abline(v=qt(0.975,18))
plot(x, y_cumul, verticals=T, do.points=F)
abline(v=qt(0.975,18))

# Compute lower and upper limit of the 95% confidence interval:

mean(age) - qt(0.975,18)*(sd(age)/sqrt(19))      # lower limit

mean(age) + qt(0.975,18)*(sd(age)/sqrt(19))     # upper limit



# Check that you get the same result as in the lectures (cf slide 18)

# Compute t-statistic:

tstat=(mean(age)-265)/(sd(age)/sqrt(19))       #t-statistic

tstat

# Add another vertical line to the CDF and the density plot at that point. Compare with the previous vertical line and comment.

plot(x, y_density, type="l")
abline(v=qt(0.975,18))
abline(v=tstat)
plot(x, y_cumul, verticals=T, do.points=F)
abline(v=qt(0.975,18))
abline(v=tstat)

# Let's do it numerically now. Compute P-value:

1-pt(tstat,18)

# Check that you get the same result as in the lectures (cf slide 22)

# What output do we expect from the command:

qt(pt(tstat,18),18)



# R has readymade commands for t-tests with corresponding confidence intervals.

# Use the command "t.test" to compute the confidence interval (this gives a two-sided test):

t.test(age,mu=265)



# Use the command "t.test" to compute a one-sided test (this gives a one-sided confidence interval):

t.test(age,alternative="greater",mu=265)



# Check that you get the same results as above. 