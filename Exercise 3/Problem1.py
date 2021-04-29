from Functions import *

#Domains
L = 100
Nz = 100001
Z, dz = np.linspace(0, L, Nz, retstep = True)
Tmax = 1e6
Nt = 1001
T, dt = np.linspace(0, Tmax, Nt+1, retstep = True)

#TEST CASE 1

K = 1e-3 + 2e-2 * Z / 7* np.exp(- Z / 7) + 5e-2 * (L - Z) / 10 * np.exp(-((L - Z) / 10))
C0 = np.ones(Nz)
kw = 0
C_eq = 5060 * 415e-6 * np.ones(Nt+1)

C = iterator(C0, Nt, dt, dz, K, kw, C_eq)

plt.title("Numerical Check of the Well-Mixed Condition")
plt.plot(Z, C[0], label = "Concentration at t = 0 s", color = 'blue')
plt.plot(Z, C[-1], label = f"Concentration at t = {Tmax} s", color = 'red')
plt.xlabel(r"z (m)")
plt.ylabel(r"C (mol m$^{-3}$)")
plt.ylim(0, 2 * C0[0])
plt.legend()
plt.show()

#TEST CASE 2
sigma1 = Nz//10
C0 = gaussian(Nz, sigma1)
#C0 = 1 / (sigma1 * np.sqrt(2*np.pi)) * np.exp(-((Z - mu1) / sigma1)**2 / 2) #Gaussian initial concentration
C = iterator(C0, Nt, dt, dz, K, kw, C_eq)

plt.plot(Z, C[0], label = "Concentration at t = 0 s", color = 'blue')
plt.plot(Z, C[-1], label = f"Concentration at t = {Tmax} s", color = 'red')
plt.xlabel(r"z (m)")
plt.ylabel(r"C (mol m$^{-3}$)")
plt.legend()
plt.show()

M = simps(C, x = Z, axis = 1)
plt.plot(T, (M - M[0])/M[0], color = "black")
plt.title("Change in Mass versus Time")
plt.ylabel(r"(M - M$_{0}$)/M$_{0}$")
plt.xlabel("Time (s)")
plt.show()

#TEST CASE 3

K = 2e-3 * np.ones(Nz) #Set constant diffusivity
C = iterator(C0, Nt, dt, dz, K, kw, C_eq)

M = simps(C, x=Z, axis = 1)
mu = simps(C * Z, x = Z, axis = 1) / M
Z_matrix = np.zeros((len(mu), len(Z)), dtype = Z.dtype) + Z #Make matrix with mu.shape copies of Z
var = simps(C * (Z_matrix - mu[:, None])**2, x = Z, axis = 1) / M

var_linear = var[0] + 2 * K[0] * T
var_steady = 2500/3 * np.ones(T.shape) #Found by taking the integral for the variance with C = const

plt.plot(T, var, label = "Observed variance", color = "black")
plt.plot(T, var_linear, '--', label = r"$\sigma_{0}^2 + 2Kt$", color = "blue")
plt.plot(T, var_steady, '--', label = r"$\sigma_{\infty}^2$", color = "red")
plt.title("Variance versus Time")
plt.ylabel(r"$\sigma^{2}$")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

#TEST CASE 4

#The following values give Bi = 0.01 for L = L = 100
K = 1e4 * np.ones(Nz)
kw = 6.97e-5
Tmax = 2e7
T, dt = np.linspace(0, Tmax, Nt+1, retstep = True)
C_eq = np.zeros(Nt+1) #Zero concentration outside mass-transfer boundary
tau = L / kw #Define decay time

C0 = np.ones(Nz)
C = iterator(C0, Nt, dt, dz, K, kw, C_eq)

M = simps(C, x = Z, axis = 1)
M_theory = M[0] * np.exp(-T / tau)

plt.plot(T, M, color = "blue")
plt.plot(T, M_theory, '--', color = "black", label = r"M$_{0} \exp$$(-t/ \tau)$")
plt.title("Rate of Mass Transfer for Constant Diffusivity")
plt.ylabel("M")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

#Variable diffusivity
K = 1e4 + 2e3*(Z/7)*np.exp(-Z/7) + 5e3*((L-Z)/10)*np.exp(-(L-Z)/10)
C = iterator(C0, Nt, dt, dz, K, kw, C_eq)

M = simps(C, x = Z, axis = 1)
M_theory = M[0] * np.exp(-T / tau)
plt.plot(T, M, color = "blue")
plt.plot(T, M_theory, '--', color = "black", label = r"M$_{0} \exp$$(-t/ \tau)$")
plt.title("Rate of Mass Transfer for Variable Diffusivity")
plt.ylabel("M")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

#TEST CASE 5
#Constant diffusivity
K = 1e4 * np.ones(Nz)
#K = 1e-2 * np.ones(Nz)
C_eq = 5060 * 415e-6 * np.ones(Nt+1)
C0 = np.zeros(Nz)
C = iterator(C0, Nt, dt, dz, K, kw, C_eq)
C_min = []
C_max = []
for i in range(len(C)):
    C_min.append(np.amin(C[i]))
    C_max.append(np.amax(C[i]))

plt.plot(T, C_eq, '--', label = r"C$_{eq}$", color = "black")
plt.plot(T, C_min, label = r"C$_{min}$", color = "blue")
plt.plot(T, C_max, label = r"C$_{max}$", color = "red", linestyle = "--")
plt.title("Equilibrium Concentration for Constant Diffusivity")
plt.ylabel(r"C (mol m$^{-3}$)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()

#Variable diffusivity
#K = 1e-2 + 2e-2 * Z / 7* np.exp(- Z / 7) + 5e-2 * (L - Z) / 10 * np.exp(-((L - Z) / 10))
K = 1e4 + 2e3 * Z / 7 * np.exp(- Z / 7) + 5e3 * (L - Z) / 10 * np.exp(-((L - Z) / 10))
C = iterator(C0, Nt, dt, dz, K, kw, C_eq)
C_min = []
C_max = []
for i in range(len(C)):
    C_min.append(np.amin(C[i]))
    C_max.append(np.amax(C[i]))

plt.plot(T, C_eq, '--', label = r"C$_{eq}$", color = "black")
plt.plot(T, C_min, label = r"C$_{min}$", color = "blue")
plt.plot(T, C_max, label = r"C$_{max}$", color = "red", linestyle = "--")
plt.title("Equilibrium Concentration for Variable Diffusivity")
plt.ylabel(r"C (mol m$^{-3}$)")
plt.xlabel("Time (s)")
plt.legend()
plt.show()



