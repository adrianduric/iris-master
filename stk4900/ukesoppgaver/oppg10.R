#Exercise 10:  multiple linear regression

# The Heart and Estrogen/Progestin Study (HERS) is a clinical trial of hormone therapy for prevention of recurrent heart attacks and death among post-menopausal women with existing coronary heart disease. The HERS data are used in many of the examples in Chapters 3 and 4 of the course text book byVittinghoff et al. In this exercise we will study how different variables may influence the glucose level in the blood for the non-diabetic women in the cohort, in particular we are interested to see if exercise may help to reduce the glucose level (cf. Section 4.1 in Vittinghoff et al.).

# You may read the HERS data into R and extract the women without diabetes by the commands,

hers=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/data/hers.txt",sep="\t",header=T,na.strings=".")
hers.no=hers[hers$diabetes==0, ]

# We will start out by investigating (in questions a-c) how the glucose levels are for women who exercise at least three times a week (coded as exercise=1) and women who exercise less than three times a week (coded as exercise=0).

# a)

# Make a summary and boxplot of the glucose levels according to the level of exercise:

summary(hers.no$glucose[hers.no$exercise==0])
summary(hers.no$glucose[hers.no$exercise==1])
boxplot(hers.no$glucose~hers.no$exercise)

 

# Discuss what the summaries and boxplot tell you.

# Glukosenivåene ser ut til å være ganske like for de som trener og ikke, når de ikke har diabetes.
 
 

# b)

# Test if there is a difference in glucose level and make a confidence interval:

t.test(glucose~exercise, var.equal=T,data=hers.no)

# What may you conclude for the test and the confidence interval?

# Det er en signifikant forskjell i gjennomsnittlig glukosenivå for dem som trener og ikke.

# c)

# Perform a simple linear regression with glucose level as outcome and exercise as predictor:

fit.c=lm(glucose~exercise,data=hers.no)
summary(fit.c)

# Discuss how the result of the simple linear regression relates to those in question b.

# Vi ser at det er estimert med høy konfidens at trening senker glukosenivået.

# The women who exercise at least three times a week and the women who exercise less than three times a week may differ in many ways. For example they may be younger and have a lower BMI (body mass index). We will therefore perform a multiple linear regression analysis where we adjust for these to variables.

# d)
# Perform a simple linear regression with glucose level as outcome and exercise, age, and BMI as predictors:

fit.d=lm(glucose~exercise+age+BMI,data=hers.no)
summary(fit.d)

# Discuss the result of this analysis in relation with the result in question c. Also discuss how age and BMI influence the glucose level.

# Vi ser at signifikansen i effekten til trening er mindre, dog fortsatt signifikant. Alder er svakt (90%) signifikant med positiv koeffisient, mens BMI er sterkt signifikant og også positiv koeffisient. Resultatene kan tyde på at noe av effekten av trening fra tidligere analyse egentlig stammer fra BMI, og at BMI og treningsnivå er korrelerte.
