# Exercise 16:  confidence interval and tests for two proportions

 

# In this exercise we will use the results from the opinion polls from Norstat from February and March 2017 to investigate the change in the support for some political parties.

 

# In February 2017 Norstat asked n1=935 individuals which party they would support if there had been election to the parliament tomorrow. Of these y1=230 would have voted H�yre.

# One month later, in March,  n2=927 persons were interviewed and y2=208 of these would have voted H�yre. 

 

# Question a)

 

# We start out by estimating the change in the support for H�yre with a 95 % confidence interval  (cf. slide 6) 

# Try to program such an interval yourself in R. A suggested solution is given at the bottom of this page. Comment on the results. 

n1=935

y1=230

p1=y1/n1

se1=sqrt(p1*(1-p1)/n1)

n2=927

y2=208

p2=y2/n2

se2=sqrt(p2*(1-p2)/n2)

se=sqrt(se1^2+se2^2)

change=p2-p1

margin=1.96*se

lower=change-margin

upper=change+margin

cbind(change,margin,lower,upper)

# Endringen er på -2.2% med en feilmargin på 3.9%. Med andre ord er ikke endringen signifikant.

# Question b)

 

# We then test the null hypothesis that the support for H�yre has not changed from February to March (cf. slide 8)

p=(y1+y2)/(n1+n2)

se0=sqrt(p*(1-p)/n1+p*(1-p)/n2)

z=(p2-p1)/se0

pval=2*(1-pnorm(abs(z)))

cbind(z,pval)

# Perform these commands and comment on the results.

# Is the null hypothesis rejected or not? How does this relate to the confidence interval computed earlier?

# Vi får ekvivalente svar til over, og ser at endringen ikke er signifikant. Vi kan ikke forkaste H0.

 

# Question c)

 

# R has a command for comparing two proportions

hoyre=matrix(c(y1,y2,n1-y1,n2-y2),nrow=2)   # give the data for H�yre in a 2x2 table (cf. slide 10)

prop.test(hoyre,correct=F)

 

# Perform these commands and check that the results agree with those obtained earlier.

# The prop.test-command give a chi squared statistic, not a z-value as we computed earlier. What is the relation between the two?

# Chi^2 = z^2 

 

# Question d)

 

# We will then take a look at the results for Senterpartiet (Sp). In February 80 of the 935  persons who were interviewed would have voted Senterpartiet; while in March 101 of the 927 interviewed would have voted Senterpartiet.

 

# Estimating the change in the support for Senterpartiet with a 95 % confidence interval.

n1=935

y1=80

p1=y1/n1

se1=sqrt(p1*(1-p1)/n1)

n2=927

y2=101

p2=y2/n2

se2=sqrt(p2*(1-p2)/n2)

se=sqrt(se1^2+se2^2)

change=p2-p1

margin=1.96*se

lower=change-margin

upper=change+margin

cbind(change,margin,lower,upper) # 2.3% +/- 2.7%

# Also test the null hypothesis that the support for Senterpartiet has not changed from February to March.

p=(y1+y2)/(n1+n2)

se0=sqrt(p*(1-p)/n1+p*(1-p)/n2)

z=(p2-p1)/se0

pval=2*(1-pnorm(abs(z)))

cbind(z,pval) 

# What are your conclusions?

# Beholder H0; insignifikant endring

 

# Solution to question a)   

n1=935

y1=230

p1=y1/n1

se1=sqrt(p1*(1-p1)/n1)

n2=927

y2=208

p2=y2/n2

se2=sqrt(p2*(1-p2)/n2)

se=sqrt(se1^2+se2^2)

change=p1-p2

margin=1.96*se

lower=change-margin

upper=change+margin

cbind(change,margin,lower,upper)

 

# Perform these commands and comment on the results. Make sure that you understand the commands!