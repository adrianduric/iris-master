# QUESTION  a)

 

# Read the data into a dataframe, give names to the variables, and inspect the data:

salinity=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/data/exer5.dat")

names(salinity)=c("salt","saltprev","trend","discharge")

salinity

 

# Check that the data correspond to those given in the exercise.

 

 

# Get an overview of the data:

summary(salinity)

plot(salinity)

 

# Make sure that you understand what the summary measures tell you!

# What do you see from the scatter plots?

# Vi ser at nåværende saltnivå er sterkt positivt korrelert til tidligere saltnivå. Det er ikke en veldig tydelig korrelasjon mellom saltnivå og trenden, men det ser ut til å være en ganske sterk negativ korrelasjon mellom salt og discharge.

  

# Do linear regression with all three covariates and inspect the results:

lmfull=lm(salt~saltprev+trend+discharge, data=salinity)

summary(lmfull)

 

# How important are each of the covariates? How does this agree with the scatter plots?

# Resultatet fra regresjonen rimer med hva vi så fra scatterplottene. Tidligere saltnivå er viktigst, men også discharge påvirker saltnivået signifikant.

 

# QUESTION  b)

 

# Extract fitted values and residuals, and give a summary of the residuals:

saltfit=lmfull$fit

saltres= lmfull$res

summary(saltres)

# Residualene er definert som y - y_hat, altså differansen mellom de sanne datapunktene og modellens prediksjon for de gitte prediktorene.

 

 

# QUESTION  c)

 

# We will make various plots of the residuals



# (i) Checking normality

# Histogram and Q-Q plot (make one plot at a time)

hist(saltres)

qqnorm(saltres); qqline(saltres)    # alternative command:  plot(lmfull,2)

 

# What do the plots tell you?

# Fra histogrammet ser vi at residualene kan se ut til å være normalfordelte, dog kanskje noe ujevnt fordelt (kanskje en venstreforskyvning). QQ-plottet viser det samme, med unntak av ytterpunktene som potensielt kan være uteliggere.

 

# (ii) Checking homoscedasticy

# Plot of residuals versus fitted values:

plot(saltfit, saltres, xlab="Fitted values", ylab="Residuals")

 

# What does the plot tell you?

# Vi kan ikke se noe tydelig mønster mellom residualene og de estimerte verdiene, som både tyder på at linearitet er rimelig. Variansen kan kanskje være varierende for ulike verdier av y; residualene ser noe mer spredt ut for lavere verdier enn for høyere.

# Alternative plots:

plot(lmfull,1)

plot(lmfull,3)

 

# What do these plots tell you?

# Det første forteller oss, som vi antok over, at normalitetsantagelsen ser ut til å være riktig. Det andre viser hvor variansen øker og synker, og viser at variansen er noe høyere for y-verdier "i midten".

 

 

# (iii) Checking linearity

# We will make CPR plots for each of the predictors

# We then use the "car" library (package)

library(car)

crPlots(lmfull, terms=~.)

 

# What do the plots tell you? Are there indications of deviation from linearity?

# (Note that the green line in the plots may be influenced quite a lot by a few observations, so one should be careful not to overinterpret the plots )  

# Kun discharge har en linje som gir mistanke om ikke-linearitet, men det er også tydelig at denne er forårsaket av en spesifikk uteligger. Uten den ser linearitetsantagelsen ut til å stemme for alle kovariatene.

 

# (iv) Studying influence

# We will make boxplots of the standardized dfbetas to check if some observations have a large influence of the estimates of the fitted model:

boxplot(dfbetas(lmfull)[,-1])

 

# You will find that one observation is very influential for the estimated effect of "discharge"

# A simple way of identifying this observations is to just list the observations together with the standardized defbetas for "discharge" (which is the 4th parameter when we also count the intercept)

cbind(salinity,dfbetas(lmfull)[,4])

 

# Which observation is the influential one?

# Nr. 16

# A more "advanced" way to identify the influential observation, is to use the identify-command (cf. the slides from the lectures).

# If time allows you may try the following commands for identifying the observation that is influential for the estimate of "discharge"

db=dfbeta(lmfull)

j=4

boxplot(db[,j])

identify(rep(1,dim(db)[1]),db[,j], labels=1:dim(db)[1])

 

 

# QUESTION d)

 

# Try fitting a model with a second degree term for discharge

lmfull2=lm(salt~saltprev+trend+discharge+I(discharge^2), data=salinity)

 

# Check the model assumptions for this model. Is it improved comare to the model without a second degree term?

saltfit2=lmfull2$fit

saltres2= lmfull2$res

hist(saltres2)
# Histogrammet er nå betydelig mer venstreforskjøvet.
qqnorm(saltres2); qqline(saltres2)
# Det ser ut til å være avvik fra normalitet blant de lavere y-verdiene. 

plot(lmfull2,1) # Ser mindre lineært ut enn før
plot(lmfull2,3) # Omtrent likt som før


# Try yourself other models (e.g. with interactions and/or more second order terms)

 

# Which model would you suggest to use for predicting salinity?

# Den første modellen virker best.