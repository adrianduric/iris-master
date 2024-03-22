pef.data = data.frame(pef = c(494,395,516,434,476,413,442,433),
                      minipef = c(512,430,520,428,500,364,380,445))
plot(pef.data$pef, pef.data$minipef)
# Ser fra plottet at det er en antydning til korrelasjon, men mye støy
# og potensielt ikke-linearitet i forholdet mellom de to.

cor(pef.data$pef, pef.data$minipef)

cor.test(pef.data$pef, pef.data$minipef)
# 95% KI: (0.2605298 0.9653948)

fit = lm(pef.data$minipef~pef.data$pef)
summary(fit)
# Minste kvadraters estimat for beta_1 = 1.1642. Med andre ord anslår
# modellen at målingen på minipef øker med 1.1642 for hver økning med 1 
# for pef. Dvs. de måler ikke nøyaktig likt (da hadde beta_1 vært lik 1).

fit$coefficients[2] * (sd(pef.data$pef) / sd(pef.data$minipef))
# De er like. Det kommer av ligningen beta^ = r*(s_y/s_x)
# --> r = beta^ * (s_x/s_y)