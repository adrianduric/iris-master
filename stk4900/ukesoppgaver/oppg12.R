# Read the data into a dataframe, give names to the variables, and inspect the data:

gun=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/v17/gun.dat", col.names=c("method","phys","team","rounds"))

gun

 

# Check that the data correspond to those given in the exercise.

  

# QUESTION a)

 

# Compute correlations:

cor(gun)

 

# How are the correlations between the covariates ("method","phys","team")?

# method-phys: 0, method-team: 0, phys-team: 0.

# Can you explain the reason for this?

# Eksperimentet var satt opp slik at det ville være like mange av hver kroppsbygning på hvert lag, og hvor en av hver kroppstype skulle bruke en av metodene. Dvs. eksperimentet var et balansert design.

# How are the correlations between the covariates and "rounds"?

# De er ikke 0, dvs. de har i eksperimentene vært målt til å ha en effekt på responsen.

 

# QUESTIONS b,c and d)

# Define the covariates as factors (categorical covariates):

gun$method=factor(gun$method)

gun$phys=factor(gun$phys)

gun$team=factor(gun$team)

 

 

# Fit a model with main effects and interactions and write the anova table:

gfit=lm(rounds~method*phys*team, data=gun) 

anova(gfit)

 

# What does the anova table tell you? Which interactions and main effects are significant?

# ANOVA-tabellen forteller oss at det er en sterkt signifikant forskjell i respons for de to ulike metodene, og svakt signifikante forskjeller i respons for forskjellige fysikknivå og forskjellige lag. Den eneste interaksjonen som er (svakt) signifikant er mellom fysikknivå og lag, som vil si at i eksperimentet hadde sammensetningen av fysikknivået og laget et individ var på en effekt på responsen til individet.

# Look at the estimates:

summary(gfit)

 

# Give an interpretation of the (most important) estimates.

# Det viktigste estimatet er beta_1 som forteller at metode 2 ga signifikant lavere respons enn metode 1. Vi ser også at lag 2 skjøt signifikant flere runder enn de to andre lagene fra beta_4, og at det var signifikant at de med fysikknivå 3 på lag 2 skjøt færre runder enn andre på laget, og andre med samme fysikknivå på andre lag.




#  For the data in R-exercise 12 on the gun data we could be interested in predicting the rounds given specified levels of method, physique and team.

# and find confidence interval for estimated expected values as well as prediction intervals for new observations given the levels of these factors.
 

# a) Set up a data frame for values where you would like to make predictions, e.g.

testdata=data.frame(method=factor(c(1,2,1,2)), phys=factor(c(1,1,2,3)), team=factor(c(1,2,3,1)))

# Then find fitted/predicted values for your favourite model gfitfav from R-exercise 12 by

gfitfav = lm(rounds~method + team*phys, data=gun)
predict(gfitfav , newdata=testdata)

 

# b)

# Then obtain confidence intervals for the expected values at this levels of the factors by

predict(gfitfav , newdata=testdata, interval="confidence")

 
# c)

# Next find the corresponding prediction intervals by

predict(gfitfav , newdata=testdata, interval="prediction")

 

# Compare and discuss the difference between the confidence and prediction intervals. 

# Konfidensintervall gir et intervall hvor den sanne gjennomsnittlige verdien for de gitte kovariatene vil befinne seg i 95% av tilfellene. Prediksjonsintervallet tar også hensyn til den individuelle støyen til en gitt prediksjon, og er derfor et intervall hvor et nytt datapunkt med de samme kovariatene vil befinne seg i 95% av tilfellene.