data = scan("data/exer2.dat")

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

