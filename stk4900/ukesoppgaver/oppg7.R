# Exercise 7:  understanding correlation



# In this exercise we will perform some simulations that may help to get intuition on the concept of correlation and the performance of the Pearson correlation coefficient. In order to keep it simple, we will assume throughout the exercise that the expected values are 0 and the standard deviations are 1.



# Before you start doing the computations, you have to load the "MASS" library into R by the command:



library(MASS)





# a) 

# Generate 25 observations (x,y) from the bivariate normal distribution with correlation 0.30 (see slide 24 from the lectures for explanation of the bivariat normal. Actual commands for generating bivariat normal data are given below)

# Compute the Pearson correlation coefficient and plot the observations:



n=25

rho=0.30

m=matrix(c(0,0),nrow=2)

S=matrix(c(1,rho,rho,1),nrow=2)

obs=mvrnorm(n,m,S)

x=obs[,1]

y=obs[,2]

cor(x,y)

plot(x,y)



# Repeat the commands a number of times. Note how the Pearson correlation coefficient and the plot vary.

# This will help to give you an intuition of how scatter plots may look like when the (true) correlation is 0.30.





# b)

# Repeat a) for correlation 0.60 and correlation 0.90. Note how the plots look like when the correlation is 0.60 and 0.90.

n=25
rho=0.60
m=matrix(c(0,0),nrow=2)
S=matrix(c(1,rho,rho,1),nrow=2)
obs=mvrnorm(n,m,S)
x=obs[,1]
y=obs[,2]
cor(x,y)
plot(x,y)

n=25
rho=0.90
m=matrix(c(0,0),nrow=2)
S=matrix(c(1,rho,rho,1),nrow=2)
obs=mvrnorm(n,m,S)
x=obs[,1]
y=obs[,2]
cor(x,y)
plot(x,y)

# c)

# Repeat a) and b) for n=100 and n=400. Note how the variation in the Pearson correlation coefficient depends on the sample size. 

n=100
rho=0.30
m=matrix(c(0,0),nrow=2)
S=matrix(c(1,rho,rho,1),nrow=2)
obs=mvrnorm(n,m,S)
x=obs[,1]
y=obs[,2]
cor(x,y)
plot(x,y)

n=100
rho=0.60
m=matrix(c(0,0),nrow=2)
S=matrix(c(1,rho,rho,1),nrow=2)
obs=mvrnorm(n,m,S)
x=obs[,1]
y=obs[,2]
cor(x,y)
plot(x,y)

n=100
rho=0.90
m=matrix(c(0,0),nrow=2)
S=matrix(c(1,rho,rho,1),nrow=2)
obs=mvrnorm(n,m,S)
x=obs[,1]
y=obs[,2]
cor(x,y)
plot(x,y)

n=400
rho=0.30
m=matrix(c(0,0),nrow=2)
S=matrix(c(1,rho,rho,1),nrow=2)
obs=mvrnorm(n,m,S)
x=obs[,1]
y=obs[,2]
cor(x,y)
plot(x,y)

n=400
rho=0.60
m=matrix(c(0,0),nrow=2)
S=matrix(c(1,rho,rho,1),nrow=2)
obs=mvrnorm(n,m,S)
x=obs[,1]
y=obs[,2]
cor(x,y)
plot(x,y)

n=400
rho=0.90
m=matrix(c(0,0),nrow=2)
S=matrix(c(1,rho,rho,1),nrow=2)
obs=mvrnorm(n,m,S)
x=obs[,1]
y=obs[,2]
cor(x,y)
plot(x,y)