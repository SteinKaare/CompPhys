from Functions import *


#Overloading the noOverlaps function to ensure that the particles are distributed in the lower half of the box
#REWRITE THIS
def noOverlaps(r, N):
    
    x = np.empty((2, N))
    x[:,0] = [0.5, 0.75]
    x[0, 1] = (1 - 2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N])
    x[1, 1] = (1/2 - 2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N])
    particleList = [(x[0, 1], x[1, 1], r[1])]
    for i in trange(2, N):
        valid = False
        while valid == False:
            p1 = ( (1 - 2 * np.amax(r)) * np.random.random() + np.amax(r) , (1/2 - 2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N]), r[i])
            j = 0
            p2 = particleList[j]
            dist = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) 
            while dist >= p1[2] + p2[2]:
                if j == len(particleList) - 1: #If we are at the final index, and the while is satisfied, the particle is valid
                    valid = True
                    particleList.append(p1)
                    x[:, i] = np.array([p1[0], p1[1]])
                    break
                j += 1
                p2 = particleList[j]
                dist = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )
               
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
    
    for k in trange(numberOfCollisions):
        if E < 0.1 * E0: return x, v, collisions, involvements
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
      
        collision = getValidCollision(collisions, involvements)
        if collision == 0: return x, v, collisions, involvements #If there are no more collisions, end the loop
        i = collision[1]
        dt = collision[0] - simTime
        E = 1/2 * m[0] * np.sum(v[:,0]**2)
        

    return x, v, collisions, involvements
    
N = 2501
r = 5*10**-3 * np.ones(N) #1 / np.sqrt(4 * np.pi * N) * np.ones(N) #Particle radii, ensures that the area taken up is 1/4
r[0] = 5 * r[0] #Set projectile radius to 5 times that of the other particles
x = noOverlaps(r, N) #Particle positions; x[0][i] is the x-coord of particle i
v = np.zeros((2, N)) #Particle velocities
v0 = 5
v[:, 0] = np.array([0, - v0]) #Give projectile speed  v0 downwards towards particle wall
m = np.ones(N) #Particle masses
m[0] = 25 * m[0] #Set projectile mass to 25 times that of the other particles

numberOfCollisions = 100000
collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
involvements = np.zeros(N)
xi = 0.5

collisions = initialisation(x, v, r, collisions, N, involvements)
first = hq.heappop(collisions)
x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions)

plt.scatter(x[0,0], x[1,0], s = 1000*r[0], c = "red")
plt.scatter(x[0, 1:], x[1, 1:], s = 1000*r[1:])
plt.show()