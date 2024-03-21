#Exercise 2:  two-sample t-test and confidence interval



#At the lectures we looked an example on bone mineral density (cf. slide 25 from the lectures)

#We will in this exercise see how the computations for this example may be done in R.





#Start by reading the data into R. This may be done by the command:

cont=c(0.228, 0.207, 0.234, 0.220, 0.217, 0.228, 0.209, 0.221, 0.204, 0.220, 0.203, 0.219, 0.218, 0.245, 0.210)

treat=c(0.250, 0.237, 0.217, 0.206, 0.247, 0.228, 0.245, 0.232, 0.267, 0.261, 0.221, 0.219, 0.232, 0.209, 0.255)

# Find the means and standard deviations from the formulas, and check that you get the same results as in the lectures (slide 25) 

cont_mean = mean(cont)
cont_sd = sd(cont)
treat_mean = mean(treat)
treat_sd = sd(treat)

# Plot both histograms and boxplots. 

par(mfrow=c(1,2))
hist(cont)
hist(treat)
dev.off()

par(mfrow=c(1,2))
boxplot(cont)
boxplot(treat)
dev.off()

# In basic R, you can show plots side by side by using the "par" command before calling the plot commands..

# For example, if we want to show 2 plots in 1 row and 2 columns the command would be: "par(mfrow=c(1,2))" You can reset the command if necessary with "dev.off()" 

# The command "density" allows us to plot a smooth version of the histogram. Given that the smoothness is created artificially, this may not be accurate when the number of samples is very low. But it may help to get a better visual interpretation of the histograms.

#Test it by using the commands (adjust the range by playing with the values of "xlim" and "ylim" inside the plot command:

plot(density(cont), col="blue", xlim=c(0.18, 0.3))

lines(density(treat),col="green")

# Add vertical lines of different color at both means.

abline(v=cont_mean, col="blue")
abline(v=treat_mean, col="green")

# Use the command "t.test" to compute the confidence interval, t-statistic and P-value:

t.test(treat, cont , var.equal=T)

# Make sure that you understand the output from the command and check that you get the same results as in the lectures (slides 27 and 28)

# Optional: Use the formulas given on slide 26 to compute the pooled standard deviation, the standard error of the effect of treatment, and the 95% confidence interval.

# Optional: Use the formulas given on slide 28 to compute the t-statistic and the P-value. 