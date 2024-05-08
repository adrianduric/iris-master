# Read the data into a dataframe and give names to the variables.
cafe = read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/data/exer3_1.dat", header=FALSE)
names(cafe)=c("no","sale")

attach(cafe)
plot(no, sale)

linfit = lm(sale ~ no)
summary(linfit)
abline(linfit, col="blue")

sqfit = lm(sale ~no + I(no^2))
summary(sqfit)
x=seq(0,7,0.1)
koef=lm(sale~no+I(no^2))$coef
koef
lines(x,koef[1]+koef[2]*x+koef[3]*x^2,lty=2, col="green")

# Vi ser at andreordens polynomet passer bedre til å beskrive forholdet enn lineært gjorde.