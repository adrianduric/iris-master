# The data are from a old study of children with leukemia.

# The children were treated so that they had no symptoms (one say that they are in remission).

# They were then given an active treatment ("Drug") to keep the symptoms from coming back or placebo ("Control").

# Time until the symptoms came back (called relapse) were measured.

# For some children the symptoms had not come back at the end of the study. These give rise to censored observations.

 

 

# Read the data into a dataframe and inspect the data:

gehan=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/data/gehan.txt", header=T)

gehan

 

# Check that the data correspond to those given in the exercise.

# "time" is time in weeks to relapse or censoring

# "cens" is 1 for relapse and 0 for censoring

# "treat" is 1 for "Control" and 2 for "Drug"

 

# Attach the R-library for survival analysis:

library(survival)

 

 

# QUESTION 1

 

# Compute Kaplan-Meier estimates for the two groups (without confidence intervals)

fit.1=survfit(Surv(time,cens)~treat, conf.type="none", data=gehan)

summary(fit.1)

 

# Make sure you understand what the output tells!

# Outputten gir KE-estimater (dvs. estimerer overlevelsesfunksjonen S(t) ved hvert tidspunkt når en ny "død" er registrert)

 

# Plot the Kaplan-Meier estimates:

plot(fit.1,lty=1:2)

 

# Interpret the plots.

# Read from the plots (approximately) what is the median time to relapse for the two groups

# Check the results by giving the command print(fit.1)

# Plottene viser KE-estimatene for de to behandlingene. Treat=1 har lavere estimert S(t) på ca. alle tidspunkter enn treat=2. Mediantiden for relapse er ca. t=8 for treat=1, og ca. t=23 for treat=2.

print(fit.1) # Riktig lest av.
 

# QUESTION 2

 

# Compute Kaplan-Meier estimates for the two groups with confidence intervals

# (the default confidence interval in R is not a good choice, so we make an explicit choice of the type of confidence interval)

fit.2=survfit(Surv(time,cens)~treat, conf.type="plain",data=gehan)

summary(fit.2)

plot(fit.2, conf.int=T, lty=1:2)

 

# Interpret the output and the plot.

# Vi ser større KI for treat=2 enn for treat=1.

 

# QUESTION 3

 

# Log-rank test for difference between the groups:

survdiff(Surv(time,cens)~treat, data=gehan)

 

# What does the output tell you?

# Chisq-verdien er høy og p-verdien lav, så vi kan forkaste H0 og fastslå med høy konfidens at S(t) er ulik mellom treat=1 og treat=2.

 

# QUESTION 4

 

# Cox regresion with treatment group as covariate:

fit.4=coxph(Surv(time,cens)~factor(treat), data=gehan)

summary(fit.4)

 

# Interpret the results!

# Cox-regresjon modellerer hasardfunksjonen h(t) per individ som en lineær prediktorfunksjon av kovariater, i dette tilfellet treat. Vi ser av den estimerte koeffisienten og KI-et at HR_2-estimatet er 0.2076 for treat=2 ift. treat=1. Dvs. hasardfunksjonen er estimert til å være ca. 5 ganger så liten for treat=2 som for treat=1.