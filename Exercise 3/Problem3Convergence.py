from Functions import *

#Convergence test for timesteps
#Domain
L = 4000
Nz_ref = 2001
Z, dz_ref = np.linspace(0, L, Nz_ref, retstep = True)
Tmax = 364 * 24 * 3600 #1 year simulation time in seconds
#Problem parameters
K = 1e-2 + (1e-4 - 1e-2) / (1 + np.exp( - 0.5 * (Z - 100)))  #Diffusivity
kw = 6.97e-5 #Mass transfer coefficient
C0 = 5060 * 415e-6 * np.ones(Nz_ref) #Initial DIC concentration

#Reference solution
dt_ref =  0.1 * 3600
Nt_ref = int(Tmax/dt_ref)
C_eq = makeC_eq(Tmax, Nt_ref) #CO2 concentration

C_ref = iteratorReturnLastTime(C0, Nt_ref, dt_ref, dz_ref, K, kw, C_eq)

M1_ref = simps(C_ref * Z, x = Z) #Reference first moment
M2_ref = simps(C_ref * Z**2, x = Z) #Reference second moment

#Test timesteps:
hourlabels = [0.5, 1, 2, 6, 12, 24, 48, 96, 168, 336, 672, 1248] #Labels for wanted hours used in plot   
hours = np.array(hourlabels) #Turn the wanted hour array into numpy array for operational ease
dts = 3600 * hours #Timesteps in seconds

M1_errorT = np.zeros(len(dts))
M2_errorT = np.zeros(len(dts))
RMS_errorT = np.zeros(len(dts))
L2_errorT = np.zeros(len(dts))
for i, dt in enumerate(dts):
    Nt = int(Tmax/dt)
    C_eq = makeC_eq(Tmax, Nt) #CO2 concentration
    C = iteratorReturnLastTime(C0, Nt, dt, dz_ref, K, kw, C_eq)
    M1_errorT[i] = np.abs(M1_ref - simps(C * Z, x = Z))
    M2_errorT[i] = np.abs(M2_ref - simps(C * Z**2, x = Z))
    RMS_errorT[i] = np.sqrt(np.mean((C - C_ref)**2))
    L2_errorT[i] = np.abs(np.linalg.norm(C - C_ref))
    
fig1, ax1 = plt.subplots()
ax1.plot(hours, M1_errorT, marker = '.' ,label = "M1", color = "blue")
ax1.plot(hours, M2_errorT, marker = '.', label = "M2", color = "red")
ax1.plot(hours, hours**2, linestyle = "--", label = r"$\sim \Delta t^2$", color = 'black')

ax1.set_xscale('log')
ax1.set_xticks(hours)
ax1.set_xticklabels(hourlabels)
ax1.set_yscale('log')
ax1.set_xlabel("Timestep (h)")
ax1.set_ylabel("Absolute Error")
ax1.tick_params('x', labelsize = 8)
plt.title("Error in Moments vs. Timestep")
plt.legend()
plt.show()

fig2, ax2 = plt.subplots()
ax2.plot(hours, RMS_errorT, marker = '.' ,label = "RMS", color = "blue")
ax2.plot(hours, L2_errorT, marker = '.' ,label = "L2", color = "red")
ax2.plot(hours, 1e-9 * hours**2, linestyle = "--", label = r"$\sim \Delta t^2$", color = 'black')
ax2.set_xscale('log')
ax2.set_xticks(hours)
ax2.set_xticklabels(hourlabels)
ax2.set_yscale('log')
ax2.set_xlabel("Timestep (h)")
ax2.set_ylabel("Absolute Error")
ax2.tick_params('x', labelsize = 8)
plt.title("Error in RMS and L2 Norm vs. Timestep")
plt.legend()
plt.show()

#Convergence test for gridspacing
#Reference solution
dt_ref = 12 * 3600
Nt_ref = int(Tmax/dt_ref)
dz_ref = 0.01
Nz_ref = int(L/dz_ref) + 1
Z_ref = np.linspace(0, L, Nz_ref)
K = 1e-2 + (1e-4 - 1e-2) / (1 + np.exp( - 0.5 * (Z_ref - 100)))
C0 = 5060 * 415e-6 * np.ones(Nz_ref) #Initial DIC concentration
C_eq = makeC_eq(Tmax, Nt_ref) #CO2 concentration
C_ref = iteratorReturnLastTime(C0, Nt_ref, dt_ref, dz_ref, K, kw, C_eq)

M1_ref = simps(C_ref * Z_ref, x = Z_ref) #Reference first moment
M2_ref = simps(C_ref * Z_ref**2, x = Z_ref) #Reference second moment

#Test gridspacing:
dzlabels = [0.05, 0.1, 0.5, 1, 2, 5]
dzs = np.array(dzlabels)
M1_errorZ = np.zeros(len(dzs))
M2_errorZ = np.zeros(len(dzs))
RMS_errorZ = np.zeros(len(dzs))
L2_errorZ = np.zeros(len(dzs))
for i, dz in enumerate(dzs):
    Nz = int(L/dz) + 1
    Z = np.linspace(0, L, Nz)
    K = K = 1e-2 + (1e-4 - 1e-2) / (1 + np.exp( - 0.5 * (Z - 100))) #Diffusivity
    C0 = 5060 * 415e-6 * np.ones(Nz)
    C = iteratorReturnLastTime(C0, Nt_ref, dt_ref, dz, K, kw, C_eq)
    M1_errorZ[i] = np.abs(M1_ref - simps(C * Z, x = Z))
    M2_errorZ[i] = np.abs(M2_ref - simps(C * Z**2, x = Z))
    Nskip = int((Nz_ref - 1)/(Nz - 1))
    RMS_errorZ[i] = np.sqrt(np.mean((C - C_ref[::Nskip])**2))
    L2_errorZ[i] = np.abs(np.linalg.norm(C- C_ref[::Nskip]))

fig3, ax3 = plt.subplots()
ax3.plot(dzs, M1_errorZ, marker = '.' ,label = "M1", color = "blue")
ax3.plot(dzs, M2_errorZ, marker = '.', label = "M2", color = "red")
ax3.plot(dzs, dzs**2, linestyle = "--", label = r"$\sim \Delta z^2$", color = 'black')
ax3.set_xscale('log')
ax3.set_xticks(dzs)
ax3.set_xticklabels(dzlabels)
ax3.set_yscale('log')
ax3.set_xlabel("Gridspacing (m)")
ax3.set_ylabel("Absolute Error")
ax3.tick_params('x', labelsize = 8)
plt.title("Error in Moments vs. Gridspacing")
plt.legend()
plt.show()

fig4, ax4 = plt.subplots()
ax4.plot(dzs, RMS_errorZ, marker = '.' ,label = "RMS", color = "blue")
ax4.plot(dzs, L2_errorZ, marker = '.' ,label = "L2", color = "red")
ax4.plot(dzs, 1e-6 * dzs**2, linestyle = "--", label = r"$\sim \Delta z^2$", color = 'black')
ax4.set_xscale('log')
ax4.set_xticks(dzs)
ax4.set_xticklabels(dzlabels)
ax4.set_yscale('log')
ax4.set_xlabel("Gridspacing (m)")
ax4.set_ylabel("Absolute Error")
ax4.tick_params('x', labelsize = 8)
plt.title("Error in RMS and L2 vs. Gridspacing")
plt.legend()
plt.show()
