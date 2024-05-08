# Exercise 15:  confidence interval for proportions

 

#At the lectures we looked an example concerning the opinion poll from February 2017 (cf. slide 2 from the lectures)

#We will in this exercise consider this example further.

 

 

 

# Question a)

 

# Of the n=935 persons who were interviewed by Norstat, y=sum(yi)=309 would have voted Ap

# The following calculations reproduce the result from the lectures (cf. slide 4)

n=935

y=309

p=y/n

se=sqrt(p*(1-p)/n)

margin=1.96*se

lower=p-margin

upper=p+margin

cbind(p,margin,lower,upper)

 

# Do the calculations and check that you get the result from the lectures.

 

 

# Question b)

 

# In the opinion poll, 122 of the persons interviewed would have voted Fremskrittspartiet (FrP) and 80 would have voted Senterpartiet (Sp).

 

# Repeat the calculations above for Fremskrittspartiet and Senterpartiet.

# How is the "margin of error" for these parties compared to the "margin of error" for Ap (cf. slide 4)?

y=122

p=y/n

se=sqrt(p*(1-p)/n)

margin=1.96*se

lower=p-margin

upper=p+margin

cbind(p,margin,lower,upper) # 13% +/- 2%


y=80

p=y/n

se=sqrt(p*(1-p)/n)

margin=1.96*se

lower=p-margin

upper=p+margin

cbind(p,margin,lower,upper) # 8.5% +/- 1.8%

# Feilmarginen for disse partiene er mindre enn for Ap på 3%.