from Functions import *
import sys
label = sys.argv[1]


#Simulation loop
#Overwriting the loop from Functions; in addition to the main features in that loop we also store the speeds in a file
@profile
def loop(x, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions):
    collision = first
    i = collision[1]
    dt = collision[0]
    simTime = 0 #The accumulated simulated time
    #Write initial speed to file
    with open(f"problem1_version{label}_v0={v0}_NOC={numberOfCollisions}_part={N}.npy", "ab") as f:
        np.save(f, np.sqrt(v[0]**2 + v[1]**2))
            
    for k in trange(numberOfCollisions):
        involvements[i] += 1
        x += v * dt
        simTime += dt
        if collision[2] == 'h' or collision[2] == 'v': #If collision is with a wall, update velocities and calculate next collisions accordingly
            v = updateVelocitiesWall(i, v, collision, xi)
            collisions = nextCollisionWall(i, x, v, r, N, collisions, involvements, simTime)
        else: #If collision is with a particle, update velocities and calculate next collisions accordingly
            j = collision[2]
            involvements[j] += 1
            v = updateVelocitiesParticles(i, j, x, v, r, m, xi)
            collisions = nextCollisionParticles(i, j, x, v, r, N, collisions, involvements, simTime)
        #Write speeds to file
        if k % 100 == 0: #Store speeds for every 100th collision
            with open(f"problem1_version{label}_v0={v0}_NOC={numberOfCollisions}_part={N}.npy", "ab") as f:
                np.save(f, np.sqrt(v[0]**2 + v[1]**2))
        collision = getValidCollision(collisions, involvements)
        if collision == 0: return x, v, collisions, involvements #If there are no more collisions, end the loop
        i = collision[1]
        dt = collision[0] - simTime

    return x, v, collisions, involvements
    
    
N = 5000
r = 1 / np.sqrt(4 * np.pi * N) * np.ones(N) #Particle radii, ensures that the area taken up is 1/4
x = noOverlaps(r,N) #Particle positions; x[0][i] is the x-coord of particle i
v0 = 1
theta = 2 * np.pi * np.random.random(N)
v = np.array([v0 * np.cos(theta), v0 * np.sin(theta)]) #Particle velocities
m = np.ones(N) #Particle masses
collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
involvements = np.zeros(N)
xi = 1
numberOfCollisions = 100000

collisions = initialisation(x, v, r, collisions, N, involvements)
first = hq.heappop(collisions)
x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions)