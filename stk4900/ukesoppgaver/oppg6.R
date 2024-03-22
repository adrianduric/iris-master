# In this exercise we will use data from the Heart and Estrogen/Progestin Study (HERS), a clinical trial of hormone therapy for prevention of recurrent heart attacks and death among post-menopausal women with existing coronary heart disease. These data are used in many of the examples in Chapters 3 and 4 of the course text book byVittinghoff et al.



# In this exercise we will study the relation between the ages of the women and their systolic blood pressure (sbp). The cohort consists of 2763 women, but in this exercise we will (as in section 3.3 of byVittinghoff et al.) only use a 10% sample from the full cohort. You may read the data into R by the command:

hers.sample=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/v17/hers.sample.txt",header=T)



# a)

# Read the data into R (cf. above) and inspect the data. Make a plot of the systolic blood pressure versus age:

plot(hers.sample$age,hers.sample$sbp)

# Discuss what the plot tells you.

# While there is a lot of variance, there seems to be hints of an increase in sbp with increased age. However, they seem only weakly correlated in this manner.

# b)

# Fit a linear regression model to the data, using systolic blood pressure as the outcome and age as the predictor:

hers.fit.b=lm(sbp~age,data=hers.sample)

summary(hers.fit.b)

# You may add the regression line to the plot from question a by the command: abline(hers.fit.b)

abline(hers.fit.b)

# Give an interpretation of the slope of the regression line (i.e. the effect of age on sbp). Does age have a significant effect on systolic blood pressure?

# As I thought in a) and as the regression line suggests, sbp seems to increase slightly with age. From the t-test of age, we see that the effect is in fact significant.

# Can you give a meaningful interpretation to the intercept?

# The intercept can be thought of as the default sbp when setting age to 0. However, this measure is probably not reliably accurate for age=0, because the dataset contains no sbp measures for babies.

# c)

# In order to get a more meaningful interpretation of the intercept, it may be useful to "center" the predictor by subtraction of its mean (or a quantity close to its mean).  Fit a linear regression using "age - 67" as predictor:

hers.fit.c=lm(sbp~I(age-67),data=hers.sample)

# Compare the least squares estimates with the ones you got in question b.

summary(hers.fit.c)
# The estimated increase in sbp per age is exactly the same, which is expected. However, the intercept estimate has increased a lot, and its SD has gone down, giving an even higher-confidence estimate (though it was already extremely high confidence before, but for a much larger confidence interval).

# How can you now interpret the intercept?

# The intercept can now be thought of as the expected sbp for a 67-year-old. For younger people, (x - 67) now becomes negative, meaning younger age indicates a decrease in sbp.

# d)

# Sometimes it may be useful to use another unit for measuring a predictor than the one given in the data file. To illustrate this point, we will here fit a linear regression model where age is measured in units of ten years:

hers.fit.d=lm(sbp~I(age/10),data=hers.sample)

# Compare the least squares estimates with the ones you got in question b.

summary(hers.fit.d)
# This led to the least square estimate and SD for beta_1 being multiplied by 10 exactly, which makes sense. The t-statistic and P-value remain the same.

# How can you now interpret the slope?

# The slope now indicates how much sbp is expected to increase for an age increase of 10 years, instead of 1 year.