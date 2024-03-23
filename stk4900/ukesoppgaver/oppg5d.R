#Exercise 5.d:



# Load the data into R with the command:

pef.data = data.frame(pef = c(494,395,516,434,476,413,442,433),
                      minipef = c(512,430,520,428,500,364,380,445))

# Fit a linear regression model with "minipef" as the outcome and "pef" as the predictor. 

fit = lm(pef.data$minipef~pef.data$pef)
summary(fit)

# Use the formulas from the slides to generate the same estimates you get in the summary of the model. That is:

# The residual standard error
rss = sum((fit$residuals - mean(fit$residuals))^2)
n = length(pef.data$pef)
p = length(fit$coefficients) - 1
df = n - p - 1
rse = sqrt(rss / df)
rse

# Multiple R-squared
tss = sum((pef.data$minipef - mean(pef.data$minipef))^2)
Rsq = 1 - (rss/tss)
Rsq

# Adjusted R-squared
Rsq_adj = 1 - ((1-Rsq)*(nrow(pef.data) - 1))/df
Rsq_adj

# Fstatistic
mss = sum((predict(fit,newdata=pef.data) - mean(pef.data$minipef))^2)
f = (mss/p)/(rss/df)
f

# p-value
1 - pf(f, p, n-p-1)
