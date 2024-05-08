# In this exercise we will use data from the Heart and Estrogen/Progestin Study (HERS), a clinical trial of hormone therapy for prevention of recurrent heart attacks and death among post-menopausal women with existing coronary heart disease.

 

# The aim of this exercise is to study how the change in low-density lipoprotein (LDL) cholesterol over the first year of the HERS study depends on the baseline value of LDL (i.e. the value at entry to the study) and use of hormone therapy (HT).

 

# You may read the HERS data into R by the command:

hers=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/v17/hers.txt",sep="\t",header=T,na.strings=".")

 

# Before we start doing our analysis we define the change in LDL, denoted LDLch:

hers$LDLch=hers$LDL1 - hers$LDL

 

#We also defined the centered LDL at baseline (by subtracting the mean value 145 mg/dL), denoted cLDL

hers$cLDL=hers$LDL-145 

 

 

# a)

# Fit a linear model with the change in LDL as the response and hormone therapy (HT) and baseline LDL (not centered) as covariates:

fit.a=lm(LDLch~HT+LDL, data=hers)

summary(fit.a)

 

# Give an interpretation of the estimates in the fitted model.

# Vi leser ut at både HT og LDL har signifikante, negative koeffisienter, som vil si at økning i HT eller LDL er estimert å senke verdien av LDLch.

 

# b)

# We then fit a model with HT and centered LDL at baseline as covariates:

fit.b=lm(LDLch~HT+cLDL, data=hers)

summary(fit.b)

 

# Compare the estimates for this model with those in question a.

# What is the interpretation of the estimate for the intercept?

# Estimatene for HT og LDL sine koeffisienter er nøyaktig de samme som før, som gir mening siden alle datapunktene bare har blitt forskjøvet, og en linje gjennomdem vil bare forskyves sammen med punktene. Dette forklarer også hvorfor estimatet for intercept har blitt lavere, nemlig at ettersom gjennomsnittlig LDL (som er positiv) er subtrahert vekk fra alle punktene i cLDL vil verdien av alle punktene være lavere, og følgelig vil intercept også være lavere.

 

# c)

# We then fit a model with interaction:

fit.c=lm(LDLch~HT+cLDL+HT:cLDL, data=hers)

summary(fit.c)

 

# Is there a significant interaction?

# Given an interpretation of the estimates in the model (cf. slide 19 of Lecture 4)

# In particular, what is the effect of baseline LDL for those with no hormone therapy?

# And what is the effect of baseline LDL for those on hormone therapy?

# Interaksjonen er signifikant. Estimatene for HT og cLDL sine koeffisienter har knapt endret seg og er fortsatt veldig signifikante. Siden interaksjonen er signifikant og estimert til å være negativ, vil det si at når både HT og cLDL blir større, har dette en ytterligere negativ effekt på responsen LDLch enn hva HT eller cLDL har hver for seg. Effekten av LDL på LDLch for dem med HT = 0 er bare beta_2 * x_2. For dem med HT = 0 er den samme effekten lik beta_2*x_2 + beta_3*x2*x3.