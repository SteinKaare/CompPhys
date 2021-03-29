from Functions import *

# PROBLEM 1 - Plots

N = 5000
numberOfCollisions = 100000
v0 = 1
m0 = 1
m = m0 * np.ones(N)

speeds = np.empty((4, numberOfCollisions//100, N)) #Array to store the speeds from files
for i in range(1, 5):
    with open(f"problem1_version{i}_v0={v0}_NOC={numberOfCollisions}_part={N}.npy", "rb") as f:
        for j in range(0, numberOfCollisions//100):
            speeds[i-1, j] = np.load(f)

#Plot initial speed distribution
plt.figure(1)
plt.hist(speeds[0, 0], bins = np.linspace(0, v0 + 1, 350), color = "b")
plt.title("Initial Speed Distribution of %d Particles" % N)
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.show()

# Use the ensemble to plot the equilibrium distribution at 100k collisions
equiSpeeds = [speeds[i, -1] for i in range(4)]
plt.figure(2)
plt.hist(equiSpeeds, bins = 75, stacked = True, density = True, color = 4*['b'])
plt.title("Equilibrium Speed Distribution of %d Particles" % N)
plt.xlabel("Speed")
plt.ylabel("Density")

# Plot the 2D Maxwell-Boltzmann velocity distribution for comparison
# Using the equipartition theorem, we find that avgEnergy = 2 * 1/2 kT = kT
a = 0
for v in equiSpeeds:
    a += 1 / (8 * N) * np.sum(m * v**2) #The average of the average kinetic energies for the four different systems
v = np.linspace(0, np.amax(equiSpeeds), N)
MB = m * v / a * np.exp(- m * v**2 / (2*a) )
plt.plot(v, MB, label = "Maxwell-Boltzmann Distribution", color = "black")
plt.legend()
plt.show()


# PROBLEM 2
m = np.concatenate((m0 * np.ones(N//2), 4 * m0 * np.ones(N//2)))

speeds = np.empty((4, numberOfCollisions//100, N))
for i in range(1, 5):
    with open(f"problem2_version{i}_v0={v0}_NOC={numberOfCollisions}_part={N}.npy", "rb") as f:
        for j in range(0, numberOfCollisions//100):
            speeds[i-1, j] = np.load(f)

#Initial distribution, m = m0
plt.figure(3)
plt.hist(speeds[0, 0, 0:N//2], bins = np.linspace(0, v0 + 1, 350), color = "b")
plt.title(f"Initial Speeds Distribution of {N//2} Particles, m = {m0}")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.show()

#Initial distribution, m = 4m0
plt.figure(3)
plt.hist(speeds[0, 0, N//2:], bins = np.linspace(0, v0 + 1, 350), color = "b")
plt.title(f"Initial Speeds Distribution of {N//2} Particles, m = {4*m0}")
plt.xlabel("Speed")
plt.ylabel("Frequency")
plt.show()

#Equilibrium speeds for m=m0 and m=4m0
equiSpeeds_m0 = [speeds[i, -1, 0:N//2] for i in range(4)] 
equiSpeeds_4m0 = [speeds[i, -1, N//2:] for i in range(4)]

#Equilibrium distribution, m = m0
plt.figure(5)
plt.hist(equiSpeeds_m0, bins = 50, stacked = True, density = True, color = 4*['b'])
plt.title(f"Equilibrium Speed Distribution of {N//2} Particles, m = {m0}")
plt.xlabel("Speed")
plt.ylabel("Density")
#Maxwell-Boltzmann
a = 0
for v in equiSpeeds_m0:
    a += 1 / (4 * N) * np.sum(m[0:N//2] * v**2) #The average of the average kinetic energies for the four different systems
v = np.linspace(0, np.amax(equiSpeeds_m0), N//2)
MB = m[0:N//2] * v / a * np.exp(- m[0:N//2] * v**2 / (2*a) )
plt.plot(v, MB, label = "Maxwell-Boltzmann Distribution", color = "black")
plt.legend()
plt.show()

#Equilibrium distribution, m = 4m0
plt.figure(6)
plt.hist(equiSpeeds_4m0, bins = 50, density = True, color = 4*['b'], stacked = True)
plt.title(f"Equilibrium Speed Distribution of {N//2} Particles, m = {4*m0}")
plt.xlabel("Speed")
plt.ylabel("Density")
#Maxwell-Boltzmann
a = 0
for v in equiSpeeds_4m0:
    a += 1 / (4 * N) * np.sum(m[N//2:] * v**2) #The average of the average kinetic energies for the four different systems
v = np.linspace(0, np.amax(equiSpeeds_4m0), N//2)
MB = m[N//2:] * v / a * np.exp(- m[N//2:] * v**2 / (2*a) )
plt.plot(v, MB, label = "Maxwell-Boltzmann Distribution", color = "black")
plt.legend()
plt.show()

#Times for plots
times_avg = np.zeros(numberOfCollisions//100)
#Average speeds
avgSpeeds_m0 = np.array([])
avgSpeeds_4m0 = np.array([])
avgSpeeds = np.array([])
#Load the times array
for i in range(1,5):
    with open(f"problem2_times_version{i}_v0={v0}_NOC={numberOfCollisions}_part={N}.npy", "rb") as f:
        times = np.load(f)
    times_avg += 1/4 * times

for i in range(numberOfCollisions//100):
    v1 = 0
    v2 = 0
    for j in range(4):
        # #Average over a single array: 1/(N/2). Average over four arrays: 1/4 -> 1/(2N)
        v1 += 1 / (2 * N) * np.sum(speeds[j, i, 0:N//2]) 
        v2 += 1 / (2 * N) * np.sum(speeds[j, i, N//2:])
    avgSpeeds_m0 = np.append(avgSpeeds_m0, v1)
    avgSpeeds_4m0 = np.append(avgSpeeds_4m0, v2)

plt.figure(7)
plt.plot(times_avg, avgSpeeds_m0, label = f"m = {m0}", color = "b")
plt.plot(times_avg, avgSpeeds_4m0, label = f"m = {4*m0}", color = "r")
plt.title(f"Average Speed")
plt.xlabel("Time")
plt.ylabel("Speed")
plt.legend()
plt.show()

#Average kinetic energies
avgKE_m0 = np.array([])
avgKE_4m0 = np.array([])

for i in range(numberOfCollisions//100):
    KE1 = 0
    KE2 = 0
    for j in range(4):
        #Average over a single array: 1/(N/2). Average over four arrays: 1/4 -> 1/(2N). Extra factor 1/2 due to KE
        KE1 += 1 / (4 * N) * np.sum(m[0:N//2] * speeds[j, i, 0:N//2]**2)
        KE2 += 1 / (4 * N) * np.sum(m[N//2:] * speeds[j, i, N//2:]**2)
    avgKE_m0 = np.append(avgKE_m0, KE1)
    avgKE_4m0 = np.append(avgKE_4m0, KE2)

plt.figure(8)
plt.plot(times_avg, avgKE_m0, label = f"m = {m0}", color = "b")
plt.plot(times_avg, avgKE_4m0, label = f"m = {4*m0}", color = "r")
plt.title(f"Average Kinetic Energy")
plt.xlabel("Time")
plt.ylabel("Kinetic Energy")
plt.legend()
plt.show()

# PROBLEM 3

Xi = [1.0, 0.9, 0.8]
times = np.empty((4, 3, numberOfCollisions//100))
KEtot = np.empty((4, 3, numberOfCollisions//100))
KEm0 = np.empty((4, 3, numberOfCollisions//100))
KE4m0 = np.empty((4, 3, numberOfCollisions//100)) 
for i in range(1,5):
    for j, xi in enumerate(Xi):
        with open(f"problem3_version{i}_times_xi={xi}_NOC={numberOfCollisions}_part={N}.npy", "rb") as f:
            times[i-1, j] = np.load(f)
        with open(f"problem3_version{i}_KEtot_xi={xi}_NOC={numberOfCollisions}_part={N}.npy", "rb") as f:
            KEtot[i-1, j] = np.load(f)
        with open(f"problem3_version{i}_KE_m0_xi={xi}_NOC={numberOfCollisions}_part={N}.npy", "rb") as f:
            KEm0[i-1, j] = np.load(f)
        with open(f"problem3_version{i}_KE_4m0_xi={xi}_NOC={numberOfCollisions}_part={N}.npy", "rb") as f:
            KE4m0[i-1, j] = np.load(f)
            
#Average time and energies over the four ensembles
times_avg = np.zeros((3, numberOfCollisions//100))
KEtot_avg = np.zeros((3, numberOfCollisions//100))
KEm0_avg = np.zeros((3, numberOfCollisions//100))
KE4m0_avg = np.zeros((3, numberOfCollisions//100))
for i in range(3):
    for j in range(4):
        times_avg[i] += 1/4 * times[j, i]
        KEtot_avg[i] += 1/4 * KEtot[j, i]
        KEm0_avg[i] += 1/4 * KEm0[j, i]
        KE4m0_avg[i] += 1/4 * KE4m0[j, i]

for i, xi in enumerate(Xi):
    plt.figure(8+i)
    plt.plot(times_avg[i], KEtot_avg[i], label = "Whole Gas", color = "black")
    plt.plot(times_avg[i], KEm0_avg[i], label = "Light Gas, m = %d" % m0, color = "b")
    plt.plot(times_avg[i], KE4m0_avg[i], label = "Heavy Gas, m = %d" % 4*m0, color = "r")
    plt.title(r"Average Kinetic Energy, $\xi$ = %.1f" % xi)
    plt.xlabel("Time")
    plt.ylabel("Kinetic Energy")
    plt.legend()
    plt.show()
    

