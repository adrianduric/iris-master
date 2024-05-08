# Read the data into a dataframe, give names to the variables, and inspect the data:

rock=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/data/rock.dat",col.names=c("area","peri","shape","perm"))

rock

 

# Check that the data correspond to those given in the exercise.

 

# Attach the dataframe

attach(rock)

 

 

# QUESTION a)

 

# Descriptive statistics, histogram and boxplot of "perm" (make one plot at a time):

summary(perm)

hist(perm)

boxplot(perm)

 

# How is the distribution of "perm"? Symetric or skewed?

# Kraftig høyreforskyvning.

 

# Try a log-transform:

summary(log(perm))

hist(log(perm))

boxplot(log(perm))

 

# How is the distribution of "log(perm)"? Symetric or skewed?

# (We choose to use "log(perm)" in question c.)

# Denne ser mer symmetrisk (og potensielt normalfordelt) ut.
 

 

# QUESTION b)

 

# Compute the correlation between the three covariates and make scatterplots of the covariates:

cor(rock[,1:3])

plot(rock[,1:3])

 

# Is there a strong correlation between some of the covariates?

# How could this affect the regression analysis?

# Det er en sterk positiv korrelasjon mellom area og peri. Det kan gjøre at de har en interaksjon, som bør sjekkes. Dersom bare en av dem inkluderes, kan konfundering potensielt også forekomme. Det er også en middels sterk negativ korrelasjon mellom peri og shape; de samme prinsippene gjelder her også.

 

 

# QUESTION c)

 

# Plot "log(perm)" versus the other covariates (make one plot at a time):

plot(area, log(perm))

plot(peri, log(perm))

plot(shape, log(perm))

 

# Which of the covariates seem to be of importance?

# Det kan se ut som om peri har en negativ korrelasjon med log(perm), og at shape har en positiv en. For area er det ikke en tydelig korrelasjon tilstede.

 

# Linear regression of �log(perm)� using all three covariates:

# (the options "x=T" and "y=T" are included because of the computations in question d)

pfit1=lm(log(perm)~area+peri+shape, x=T, y=T)

summary(pfit1)

 

# Which of the covariates seem to be of importance?

# Is this in agreement with what you could see from the scatterplots?

# area og peri er sterkt signifikante, mens shape er insignifikant. Dette stemmer ikke helt overens med scatterplottene; med det sagt virket det fra scatterplottene ikke som at korrelasjonen kom til å være veldig sterk med responsen for noen av kovariatene.

 

 

# QUESTION d)

 

# According to their t-values the order of the covariates is: peri > area > shape

 

# Fit simpler regression models (according to the given order):

pfit2=lm(log(perm)~area+peri, x=T, y=T)

pfit3=lm(log(perm)~peri, x=T, y=T)

 

 

# Compute cross-validated R2:
cv.R2=function(lmobj,y=lmobj$y,x=lmobj$x)
{
   a=t(x)%*%x
   d=diag(1/eigen(a)$values)
  e=eigen(a)$vector
   inv.a=e %*% d %*% t(e)
   v=diag(x%*%inv.a%*%t(x))
   SSkryss=sum((lmobj$res/(1-v))^2)
   SStot=sum((y-mean(y))^2)
   R2kryss=1-SSkryss/SStot
   R2kryss
}

cv.R2(pfit3)

cv.R2(pfit2)

cv.R2(pfit1)

 

# Which of the three models give the best prediction?

# Compare the cross-validated R2 with R2 from question c. What do you see?

# pfit2 gir høyest kryssvalidert R2, så jeg ville valgt den.

 

 

# QUESTION e)

 

# Fit a model with second order terms and interactions:

pfit1=lm(log(perm)~area+peri+shape+I(area^2)+I(peri^2)+I(shape^2)+area:peri+area:shape+peri:shape, x=T, y=T)

summary(pfit1)

 

# Which variables seem to be most important?

# According to their t-values the variables are ordered as: area:peri > area^2 > peri > area > peri^2 > area:shape > peri:shape > shape^2 > shape

 

# Fit models according to this ordering:

pfit2=lm(log(perm)~area+peri+I(area^2)+I(peri^2)+I(shape^2)+area:peri+area:shape+peri:shape,x=T,y=T)

pfit3=lm(log(perm)~area+peri+I(area^2)+I(peri^2)+area:peri+area:shape+peri:shape,x=T,y=T)

pfit4=lm(log(perm)~area+peri+I(area^2)+I(peri^2)+area:peri+area:shape,x=T,y=T)

pfit5=lm(log(perm)~area+peri+I(area^2)+I(peri^2)+area:peri,x=T,y=T)

pfit6=lm(log(perm)~area+peri+I(area^2)+area:peri,x=T,y=T)

pfit7=lm(log(perm)~peri+I(area^2)+area:peri,x=T,y=T)

pfit8=lm(log(perm)~I(area^2)+area:peri,x=T,y=T)

pfit9=lm(log(perm)~area:peri,x=T,y=T)

 

# Compute cross-validated R2 for the models:

cv.R2(pfit9)

cv.R2(pfit8)

cv.R2(pfit7)

cv.R2(pfit6)

cv.R2(pfit5)

cv.R2(pfit4)

cv.R2(pfit3)

cv.R2(pfit2)

cv.R2(pfit1)

 

 

# Which model gives the best prediction?

# Den beste modellen er nå pfit5, som inneholder area, peri, deres kvadrater og interaksjonsterm.

 

 

# QUESTION f)

 

# Use the same "recipe" as in question e

# Which model gives the best prediction? Is it the same one as in question e?

pfit2=lm(log(perm)~area+peri+I(area^2)+I(peri^2)+I(shape^2)+area:peri+area:shape+peri:shape,x=T,y=T)

pfit3=lm(log(perm)~area+peri+I(area^2)+I(peri^2)+I(shape^2)+area:peri+area:shape,x=T,y=T)

pfit4=lm(log(perm)~area+peri+I(area^2)+I(peri^2)+I(shape^2)+area:peri,x=T,y=T)

pfit5=lm(log(perm)~area+peri+I(area^2)+I(peri^2)+area:peri,x=T,y=T)

pfit6=lm(log(perm)~area+peri+I(area^2)+area:peri,x=T,y=T)

pfit7=lm(log(perm)~area+peri+area:peri,x=T,y=T)

pfit8=lm(log(perm)~area+peri,x=T,y=T)

pfit9=lm(log(perm)~peri,x=T,y=T)

cv.R2(pfit9)

cv.R2(pfit8)

cv.R2(pfit7)

cv.R2(pfit6)

cv.R2(pfit5)

cv.R2(pfit4)

cv.R2(pfit3)

cv.R2(pfit2)

cv.R2(pfit1)

# Høyest R2 kryssvalidert er pfit5, som er samme modell som sist i dette tilfellet.
 
# QUESTION g)



# Make various plots of the residuals.

# See previous exercise for help with R-commands.

plot(pfit5,1) # Ca. lineær, muligens ikke samme varians for alle y
plot(pfit5,2) # Residualenes feilledd er normalfordelte
plot(pfit5,3) # Lavere varians i residualene til høyere y-verdier (heteroskedastisitet)
