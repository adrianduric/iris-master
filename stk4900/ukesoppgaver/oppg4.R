solvents=read.table("http://www.uio.no/studier/emner/matnat/math/STK4900/data/solvents.txt",header=T)
# Dataene er i to kolonner, en for rate (respons) og en for type (prediktor).

boxplot(rate~type,data=solvents)
# Type 1 og type 2 har ganske like medianrater, men med betydelig større
# varians for type 2. Type 3 har betydelig lavere median.

solvents$type = factor(solvents$type)
aov_solvents = aov(rate~type, data=solvents)
summary(aov_solvents)
# Første rad gir frihetsgrader (K - 1), sum of squares og MSS for modellen
# (her: per type), samt F-statistikken og P-verdi fra hypotesetest for den.
# Andre rad gir frihetsgrader (n - K), sum of squares og RSS for residualene
# (her: per måling).

# Fra F-scoren ser vi at MSS >> RSS, så vi får stor F-score og veldig lav
# sannsynlighet for å observere så høy F. Med andre ord forkastes H_0 hvor
# det ble antatt at mu for alle typene var like, og vi konkluderer at det
# er signifikant forskjell mellom snittverdiene til typene.