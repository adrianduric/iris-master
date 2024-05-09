# Read the data into a dataframe, give names to the variables, and inspect the data:

cancerdata=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/data/cancer.txt")

names(cancerdata)=c("age","cig","pyr","cancer")

cancerdata

 

# Make sure that the data are the same as given in the exercise.

 

 

 

# Questions 1 & 2)

 

# We first consider the model  E(Y) = n*exp{b0+b1*s+b2*a}, where

#      Y=number of cancer cases (=cancer),

#       n=number of person years (= pyr),

#       s=number of cigarettes smoked per day (=cig)

#       a = age in years (=age)

# We may write the model on the form  E(Y)= exp{1*log(n)+b0+b1*s+b2*a}.

# Note that  log(n)  appears as a sort of "covariate" where we know that the regression coefficient takes the value 1. This is called an OFFSET.

 

# We fit the model and look at the result::

cancerfit.1=glm(cancer~offset(log(pyr))+age+cig, data=cancerdata, family=poisson)

summary(cancerfit.1)

 

# Make sure that you understand what the output tells you!.

# Are there significant effects of age and the number of cigarettes smoked?

# Ja.

# It is common to report the results of a Poisson regression by means of the rate ratio RR = exp(beta)  with confidence limits.  

# To this end we may use the function expcoef from exercise 18.

# Use the function to compute rate ratios for age and number of cigarettes:

expcoef(cancerfit.1)

 

# Give an interpretation of what the table tells you about the effect of age and the number of cigarettes smoked

# Tabellen gir RR, dvs. estimert endring i rate når enten age eller cig øker med 1 og andre kovariater holdes stille, med 95% KI. Her ser vi at RR for begge er større enn 1, f.eks. er raten estimert å bli 1.118 ganger så stor med en økning av 1 i age. Siden 95% KI ikke inneholder noe <1, er det signifikant at age og cig øker raten til responsen (lungekreft).

 

 

# QUESTION 3)

 

# We then look at a model with second order terms and interaction:

cancerfit.3=glm(cancer~offset(log(pyr))+ age+I(age^2)+cig+I(cig^2)+age:cig, data=cancerdata, family=poisson)
summary(cancerfit.3)
 

# Reduce the model by (step-wise) eliminating non-significant covariates.

# (Use Wald tests from the summary-command and/or deviances from the anova-command.)

cancerfit.4=glm(cancer~offset(log(pyr))+ age+I(age^2)+cig+I(cig^2), data=cancerdata, family=poisson)
summary(cancerfit.4)

# Discuss the interpretation of your "final model".

# Vi ser at både age og cig er signifikante i 1. og 2. orden, men koeffisientene har motsatt fortegn for 1. og 2. orden (1. orden øker risiko, mens 2. orden minker risiko). Rent matematisk kan dette tolkes som at økningen i rate som forårsakes av å øke alder eller ant. sigaretter per dag blir mindre jo høyere kovariaten allerede er. F.eks. øker risikoen for lungekreft mer om man går fra 35 til 40 år, enn om man går fra 60 til 65 år. Det samme gjelder for økning i risiko når man går fra 1 til 2 sigaretter ift. 10 til 11 sigaretter.

 

 

# ADDITIONAL QUESTION:

# Age and the number of cigarettes smoked are reported in intervals.

# We may alternatively consider these covariates as categorical.

# Such a model is fitted by the command:

cancerfit.a=glm(cancer~offset(log(pyr))+factor(age)+factor(cig), data=cancerdata, family=poisson)
summary(cancerfit.a)
 

# Give an interpretation of this model.

# Discuss how the model may be used to assess the fit of your "final model" from question 3.

# Fra denne modellen ser vi først at signifikansen er lav for lave mengder sigaretter og lav alder, dvs. vi er ikke like sikre på at de tidlige hoppene i alder og sigaretter øker risikoen for lungekreft som vi er for de neste hoppene. I tillegg kan det se ut som fra estimatene at vår antagelse om mindre økning fra gruppe til gruppe for høyere aldere/sigarettmengder er riktig, i hvert fall for sigaretter. Det er f.eks. større økning mellom 0 og 5.2 sigaretter enn mellom 27.4 og 40.8 sigaretter. For alder er det samtidig et betydelig hopp mellom 47-52, med et veldig lite hopp mellom 52-57, som ikke rimer helt med antagelsen, siden hoppene igjen blir større etter 57.