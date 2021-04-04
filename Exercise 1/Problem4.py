from Functions import *


#Overloading the noOverlaps function to ensure that the particles are distributed in the lower half of the box
def noOverlaps(r, N):
    x = np.empty((2, N))
    x[:,0] = [0.5, 0.75]
    x[0, 1] = np.random.uniform(0 + np.amax(r[1:N]), 1 - np.amax(r[1:N]))
    x[1, 1] = np.random.uniform(0 + np.amax(r[1:N]), 1/2 - np.amax(r[1:N]))
    for i in trange(2, N):
        valid = False
        while valid == False:
            #Need to pass how many particles have been placed to the function isValid
            x_new = np.random.uniform(0 + np.amax(r[1:N]), 1 - np.amax(r[1:N]))
            y_new = np.random.uniform(0 + np.amax(r[1:N]), 1/2 - np.amax(r[1:N]))
            r_new = r[i]
            valid = checkValidPos(x_new, y_new, r_new, x, r, i)
        x[:, i] = np.array([x_new, y_new])
    return x

#Simulation loop
#Overwriting the loop from Functions; in addition to the main features in that loop we also store the speeds in a file

def loop(x, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions):
    collision = first
    i = collision[1]
    dt = collision[0]
    simTime = 0 #The accumulated simulated time
    #Projectile energy
    E0 = 1/2 * m[0] * np.sum(v[:,0]**2) #Initial energy for comparison
    E = 1/2 * m[0] * np.sum(v[:,0]**2) #Energy to be updated
    energies = [E0]
    
    for k in trange(numberOfCollisions):
        if E < 0.1 * E0: return x, v, collisions, involvements, energies
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
      
        collision = getValidCollision(collisions, involvements)
        if collision == 0: return x, v, collisions, involvements #If there are no more collisions, end the loop
        i = collision[1]
        dt = collision[0] - simTime
        E = 1/2 * m[0] * np.sum(v[:,0]**2)
        energies.append(E)
    energies = np.array(energies)
    return x, v, collisions, involvements, energies
    
def speedSweep(r, m, N, file, numberOfCollisions, xi):
    m[0] = 25 * m[0] #Set projectile mass to 25 times that of the other particles
    speeds = np.linspace(5, 25, 10)
    craterSizes = np.array([])
    for v0 in speeds:
        with open(file, "rb") as f: #Get the initial positions
            x0 = np.load(f)
        
        v = np.zeros((2, N)) #Particle velocities
        v[:, 0] = np.array([0, - v0]) #Give projectile speed  v0 downwards towards particle wall
    
        collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
        involvements = np.zeros(N)
    
        collisions = initialisation(x0, v, r, collisions, N, involvements)
        first = hq.heappop(collisions)
        x, v, collisions, involvements, energies = loop(x0, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions)
        with open("problem4_EnergiesVsCollisionsSpeeds.npy", "ab") as f:
            np.save(f, energies)
        with open(file, "rb") as f: #Get the initial positions for comparison
            x0 = np.load(f)
    
        delta_x = x - x0
        crater = np.count_nonzero(delta_x[1:N]) / (N - 1) #Define crater as displaced/total
        craterSizes = np.append(craterSizes, crater)
    
    with open("problem4_speedSweep.npy", "wb") as f:
        np.save(f, speeds)
        np.save(f, craterSizes)

def massSweep(r, m, N, file, numberOfCollisions, xi):
    masses = np.linspace(25, 50, 10)
    craterSizes = np.array([])
    for m0 in masses:
        with open(file, "rb") as f: #Get the initial positions
            x0 = np.load(f)
        m = np.ones(N)
        m[0] = m0 * m[0]
        v = np.zeros((2, N))
        v[:, 0] = np.array([0, - 5]) #Set v0 = 5 for the projectile
        
        collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
        involvements = np.zeros(N)
    
        collisions = initialisation(x0, v, r, collisions, N, involvements)
        first = hq.heappop(collisions)
        x, v, collisions, involvements, energies = loop(x0, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions)
        with open("problem4_EnergiesVsCollisionsMasses.npy", "ab") as f:
            np.save(f, energies)
        with open(file, "rb") as f: #Get the initial positions for comparison
            x0 = np.load(f)
    
        delta_x = x - x0
        crater = np.count_nonzero(delta_x[1:N]) / (N - 1) #Define crater as displaced/total
        craterSizes = np.append(craterSizes, crater)
    with open("problem4_massSweep.npy", "wb") as f:
        np.save(f, masses)
        np.save(f, craterSizes)
        
N = 2501
numberOfCollisions = 100000
xi = 0.5
m = np.ones(N) #Particle masses; set to correct values in the functions called below
r = 1 / np.sqrt(4 * np.pi * N) * np.ones(N) #Particle radii, ensures that the area taken up is 1/4
r[0] = 5 * r[0] #Set projectile radius to 5 times that of the other particles
x0 = noOverlaps(r, N) #Particle positions; x[0][i] is the x-coord of particle i
with open("problem4_initPositions.npy", "wb") as f: #Save initial positions: ensures no overwrites as loop modifies x0 otherwise
    np.save(f, x0)
  
speedSweep(r, m, N, "problem4_initPositions.npy", numberOfCollisions, xi)
massSweep(r, m, N, "problem4_initPositions.npy", numberOfCollisions, xi)


