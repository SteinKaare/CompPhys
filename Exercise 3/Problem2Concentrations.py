from Functions import *

#Domain
L = 100
Nz = 10001 #Ensures dz = 0.01
Z, dz = np.linspace(0, L, Nz, retstep = True)
Tmax = 180 * 24 * 3600 #180 day simulation time in seconds
Nt = 43200
T, dt = np.linspace(0, Tmax, Nt + 1, retstep = True) #Ensures dt = 0.1h
print(dt, dz)
#Problem parameters
K = 1e-3 + 2e-2 * Z / 7 * np.exp(- Z / 7) + 5e-2 * (L - Z) / 10 * np.exp(-((L - Z) / 10)) #Diffusivity
kw = 6.97e-5 #Mass transfer coefficient
C0 = np.zeros(Nz) #Initial DIC concentration
C_eq = 5060 * 415e-6 * np.ones(Nt + 1) #CO2 concentration

#Concentration after 180 days
C = iterator(C0, Nt, dt, dz, K, kw, C_eq)
#Plot minimum and maximum concentrations
C_min = []
C_max = []
for i in range(len(C)):
    C_min.append(np.amin(C[i]))
    C_max.append(np.amax(C[i]))

plt.figure(1)
plt.title("")
plt.plot(T, C_eq, '--', label = r"C$_{eq}$", color = "black")
plt.plot(T, C_min, label = r"C$_{min}$", color = "blue")
plt.plot(T, C_max, label = r"C$_{max}$", color = "red")
plt.legend()
plt.show()

#Concentration as a function of depth for selected days
days = np.array([1, 10, 25, 50, 75, 150, 180]) * 24 * 3600 #Convert the days we want to plot C for to seconds
indices = (days/dt).astype(int) #Get the indices that correspond (approximately) to these times
plt.figure(2)
plt.title(r"Evolution of DIC Concentration in the Ocean")
for i, index in enumerate(indices):
    plt.plot(Z, C[index], label = f"Day {days[i]/(24*3600):.0f}")
plt.xlabel(r"z (m)")
plt.ylabel(r"C (mol m$^{-3}$)")
plt.legend(loc='upper right')
plt.show()


