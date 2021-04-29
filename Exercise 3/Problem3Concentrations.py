from Functions import *

#Domain
L = 4000
Nz = 40001 #Ensures dz= 0.1
Z, dz = np.linspace(0, L, Nz, retstep = True)
Tmax = 10 * 364 * 24 * 3600 #10 year simulation time in secondsNt = 43200
Nt = 7280
T, dt = np.linspace(0, Tmax, Nt + 1, retstep = True) #Ensures dt = 12h
A = 360e12 #The area of the ocean

#Problem parameters
K = 1e-2 + (1e-4 - 1e-2) / (1 + np.exp( - 0.5 * (Z - 100)))  #Diffusivity
kw = 6.97e-5 #Mass transfer coefficient
C0 = 5060 * 415e-6 * np.ones(Nz) #Initial DIC concentration
C_eq = makeC_eq(Tmax, Nt)

C = iterator(C0, Nt, dt, dz, K, kw, C_eq)

#Concentration as a function of depth for selected days
years = np.array([0, 2.5, 5, 10]) * 364 * 24 * 3600 #Convert the years we want to plot C for to seconds
indices = (years/dt).astype(int) #Get the indices that correspond (approximately) to these times

plt.figure(1)
plt.title(r"Evolution of DIC Concentration in the Ocean")
for i, index in enumerate(indices):
    if i == 1:
        plt.plot(Z, C[index], label = f"{years[i]/(364*24*3600):.1f} years")
    else:
        plt.plot(Z, C[index], label = f"{years[i]/(364*24*3600):.0f} years")
plt.xlabel(r"z (m)")
plt.ylabel(r"C (mol m$^{-3}$)")
plt.legend(loc='upper right')
plt.show()

#Mass as a function of time
M = A * simps(C, x = Z, axis = 1) * 12 #The mass in units of grams

plt.plot(T/(364*24*3600), M, color = "blue")
plt.title("Total Mass of DIC vs. Time")
plt.ylabel("M (g)")
plt.xlabel("Time (yrs)")
plt.show()

#CO2 absorption per year
years = np.arange(11) * 364 * 24 * 3600
indices = (years/dt).astype(int)
absorbedYearly = np.zeros(10)
for i in range(len(absorbedYearly)):
    absorbedYearly[i] = M[indices[i+1]] - M[indices[i]]
carbonRate = np.mean(absorbedYearly)
print("The amount of CO2 absorbed by the ocean per year is", "{:.2e}".format(carbonRate), "grams.")