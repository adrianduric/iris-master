# R-help to Exercise 26

 

 

 

# QUESTION a)

 

# Read the data and variable names into a data frame, and take a look at the data

aserum=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/data/serum.txt", header=T)

aserum

 

# Check that the data are the same as given in the table in the exercise

# Make sure that you understand how the data are coded.

  

 

# QUESTION a)

 

# Plot serum response against time for the different persons (identified by colors) and treatments

# (identified by solid or dotted line) using the matplot command for multiple lines in a plot.

hours.mat= matrix(c(1,2,3,6), nrow=4,ncol=10)

druga=matrix(aserum$serum[aserum$drug==1], nrow=4)

drugb=matrix(aserum$serum[aserum$drug==2], nrow=4)

serum.mat=cbind(druga, drugb)

matplot(hours.mat, serum.mat, type="l", lty=c(1,1,1,1,1,2,2,2,2,2), col=c(1,2,3,4,5,1,2,3,4,5),xlab="Hours",ylab="Serum",lwd=2)

 

# Think about what the plot tells you!

# Plottet forteller om serumnivå til forskjellige tider hos de forskjellige pasientene.

 

# QUESTION c)

# We first compute the AUC for each person and each drug:

auc=matrix(0,nrow=5,ncol=2)

for (i in 1:5)

for (j in 1:2)

{

s=aserum$serum[aserum$subject==i&aserum$drug==j]

auc[i,j]=s[1]+s[2]+2*s[3]+1.5*s[4]

}

 

# Perform the computations and make sure that you get the AUCs

 

# Perform a paired t-test

t.test(auc[,1],auc[,2], paired=T)

 

# What you may conclude from this hypothesis testing?

# Det er ikke en signifikant forskjell i AUC mellom de to stoffene.

 

 

# QUESTION d)

# In order to fit a random effects model we will use the nlme-package.

# If this has not been installed, you will have to do so. At the end of the introduction to R, it is described how you may install new R-packages.

# In order to fit the modell, you then give the commands:

library(nlme)        # load the nlme-package

fit.random=lme(serum~factor(drug)+factor(time),random=~1|subject,data=aserum,method="ML")

summary(fit.random)

 

 

# QUESTION e)

# In order to test if drug has an effect, you may fit a modell without drug and use the anova-command to compare the modell with the one from question d.