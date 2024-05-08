insurance=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/data/exer3_2.dat")
names(insurance)=c("income","riskave","amount")
insurance
attach(insurance)

plot(income, amount)
plot(riskave, amount)

# Plottene ser ut til å vise en sterkere korrelasjon (positiv) mellom income og amount, enn mellom income og riskave (også positiv)

cor(amount, riskave) # 0.39
cor(amount, income) # 0.91

fit1 = lm(amount ~ riskave)
summary(fit1)
plot(income, riskave)
abline(fit1)

fit2 = lm(amount ~ income)
summary(fit2)
plot(income, amount)
abline(fit2)