data = scan("http://www.uio.no/studier/emner/matnat/math/STK4900/v17/exer2.dat")

# Plotting data
par(mfrow=c(1, 3))
hist(data)
boxplot(data)
plot(ecdf(data), verticals=T, do.points=F)
dev.off()

# Dataene er venstreforskjøvet. Det finnes to uteliggere rundt 0 og -40.
# Uten uteliggerne hadde fordelingen kansje sett mer balansert ut.

mean(data)
median(data)

# Gjennomsnittet er den gjennomsnittlige verdien av alle datapunktene.
# Medianen er den midterste verdien i datasettet.

sd(data)
IQR(data)

# Standardavvik forteller om forventet avvik for et datapunkt i datasettet,
# fra gjennomsnittsverdien i datasettet.
# IQR (inter-quartile range) gir lengden på spennet fra starten av andre
# kvartil til slutten av tredje, dvs. IQR = Q_3 - Q_1.

result = t.test(data)
result

# 95% KI: (23.57059 28.85365)
# Gitt at den sanne gjennomsnittsverdien i populasjonen dataene er fra er
# lik 33.02, vil det si at dette datasettet er eksempel på de siste 5% av
# tilfellene hvor konfidensintervallet ikke inneholder sann mu.