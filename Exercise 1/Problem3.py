from Functions import *
import sys
label = sys.argv[1]
xi = float(sys.argv[2])

#Simulation loop
#Overwriting the loop from Functions; in addition to the main features in that loop we also store the speeds in a file

def loop(x, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions):
    collision = first
    i = collision[1]
    dt = collision[0]
    simTime = 0 #The accumulated simulated time
    #Store initial KEs in arrays
    times = np.array([])
    KEtot = np.array([])
    KEm0 = np.array([])
    KE4m0 = np.array([])
    
    times = np.append(times, simTime)
    KEtot = np.append(KEtot, 1/ (2 * N) * np.sum (m * (v[0]**2 + v[1]**2)))
    KEm0 = np.append(KEm0,1/ (N) * np.sum (m[0:N//2] * (v[0, 0:N//2]**2 + v[1, 0:N//2]**2))) #1/N since 1/2 * 1/(2N)
    KE4m0 = np.append(KE4m0, 1/ (N) * np.sum (m[N//2:] * (v[0, N//2:]**2 + v[1, N//2:]**2))) #1/N since 1/2 * 1/(2N)
        
    for k in trange(numberOfCollisions):
        involvements[i] += 1
        x += v * dt
        simTime += dt
        if collision[2] == -1 or collision[2] == -2: #If collision is with a wall, update velocities and calculate next collisions accordingly
            v = updateVelocitiesWall(i, v, collision, xi)
            collisions = nextCollisionWall(i, x, v, r, N, collisions, involvements, simTime)
        else: #If collision is with a particle, update velocities and calculate next collisions accordingly
            j = collision[2]
            involvements[j] += 1
            v = updateVelocitiesParticles(i, j, x, v, r, m, xi)
            collisions = nextCollisionParticles(i, j, x, v, r, N, collisions, involvements, simTime)
        #Write KEs to files
        if k != 0 and k % 100 == 0: #Store for every 100th collision
              times = np.append(times, simTime)
              KEtot = np.append(KEtot, 1/ (2 * N) * np.sum (m * (v[0]**2 + v[1]**2)))
              KEm0 = np.append(KEm0, 1 / (N) * np.sum (m[0:N//2] * (v[0, 0:N//2]**2 + v[1, 0:N//2]**2))) #1/N since 1/2 * 1/(2N)
              KE4m0 = np.append(KE4m0, 1/ (N) * np.sum (m[N//2:] * (v[0, N//2:]**2 + v[1, N//2:]**2))) #1/N since 1/2 * 1/(2N)
        collision = getValidCollision(collisions, involvements)
        if collision == 0: return x, v, collisions, involvements #If there are no more collisions, end the loop
        i = collision[1]
        dt = collision[0] - simTime
    #Store data to files
    with open(f"problem3_version{label}_times_xi={xi}_NOC={numberOfCollisions}_part={N}.npy", "ab") as f:
        np.save(f, times)
    with open(f"problem3_version{label}_KEtot_xi={xi}_NOC={numberOfCollisions}_part={N}.npy", "ab") as f:
        np.save(f, KEtot)
    with open(f"problem3_version{label}_KE_m0_xi={xi}_NOC={numberOfCollisions}_part={N}.npy", "ab") as f:
        np.save(f, KEm0)
    with open(f"problem3_version{label}_KE_4m0_xi={xi}_NOC={numberOfCollisions}_part={N}.npy", "ab") as f:
        np.save(f, KE4m0)

    return x, v, collisions, involvements
    
N = 5000
r = 1 / np.sqrt(4 * np.pi * N) * np.ones(N) #Particle radii, ensures that the area taken up is 1/4
x = noOverlaps(r,N) #Particle positions; x[0][i] is the x-coord of particle i
v0 = 1
theta = 2 * np.pi * np.random.random(N)
v = np.array([v0 * np.cos(theta), v0 * np.sin(theta)]) #Particle velocities
m0 = 1 #Set m0 to the same value as in Problem 1
m = np.concatenate((m0 * np.ones(N//2), 4 * m0 * np.ones(N//2)))
collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
involvements = np.zeros(N)

numberOfCollisions = 100000
collisions = initialisation(x, v, r, collisions, N, involvements)
first = hq.heappop(collisions)
x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions)
