from Functions import *
import time

def oneParticleTests():
    N = 1 #Number of particles in the gas
    x = np.zeros((2, N)) #Particle positions; x[0][i] is the x-coord of particle i
    v = np.zeros((2, N)) #Particle velocities
    r = np.array([0.001]) #Particle radii
    m = np.array([1]) #Particle masses
    collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
    involvements = np.zeros(N)

    #Check that for xi=1 a particle bounces from straight on wall collision with same speed in opposite direction
    print("Test 1: Head on against wall for xi = 1" )
    xi = 1
    x[:,0] = [0.5, 0.5]
    v[:,0] = [0.5, 0]
    print("Initial velocity: ", v)
    collisions = initialisation(x, v, r, collisions, N, involvements)
    first = hq.heappop(collisions)
    x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, 1)
    print("Final velocity: ", v)

    #Law of reflection
    collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
    involvements = np.zeros(N)
    print("Test 2: Law of reflection")
    wallVector = np.array([0, 1])
    x[:,0] = [0.5, 0.75]
    v[:,0] = [0.25, -0.25]
    cosine = np.dot(wallVector, v[:,0]) / (np.linalg.norm(v[:,0]) * np.linalg.norm(wallVector))
    print("Initial cosine of angle: ", cosine)
    collisions = initialisation(x, v, r, collisions, N, involvements)
    first = hq.heappop(collisions)
    x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, 1)
    cosine = np.dot(wallVector, v[:,0]) / (np.linalg.norm(v[:,0]) * np.linalg.norm(wallVector))
    print("Final cosine of angle: ", cosine)

    #[v0, v0] strikes all walls before returning to initial positions
    collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
    involvements = np.zeros(N)
    print("Test 3: [v0, v0] strikes all walls before returning to initial position")
    x[:,0] = [0.5, 0.5]
    v[:,0] = [0.25, 0.25]
    v0 = [0.25, 0.25]
    collisions = initialisation(x, v, r, collisions, N, involvements)
    first = hq.heappop(collisions)
    x, v, collisions, involvements =loop(x, v, r, m, xi, N, first, collisions, involvements, 4) #Setting the number of collisions to four means one collision with each wall
    cosine = np.dot(v[:,0], v0) / (np.linalg.norm(v[:,0]) * np.linalg.norm(v0)) #Calculate cosine of angle of initial and final velocity
    print("Expected cosine of initial and final velocity: 1")
    print("Calculated cosine of initial and final velocity: ", cosine)

    #Xi=0: Particle stops at wall
    print("Test 4: Particle stops at wall for xi = 0")
    involvements = np.zeros(N)
    xi = 0
    x[:,0] = [0.5, 0.5]
    v[:,0] = [0.5, 0]
    print("Initial velocity: ", v)
    collisions = initialisation(x, v, r, collisions, N, involvements)
    first = hq.heappop(collisions)
    x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, 1)
    print("Final velocity: ", v)

def twoParticleTests():
    N = 2 #Number of particles in the gas
    x = np.zeros((2, N)) #Particle positions; x[0][i] is the x-coord of particle i
    v = np.zeros((2, N)) #Particle velocities
    r = np.array([0.001, 0.001]) #Particle radii
    m = np.array([1, 1]) #Particle masses
    collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
    involvements = np.zeros(N)

    #Two identical particles bounce back when they hit each other (xi=1)
    print("Test 1: Identical particles bounce back for xi = 1")
    xi = 1
    x[:, 0] = [0.25, 0.5]
    x[:, 1] = [0.75, 0.5]
    v[:, 0] = [0.5, 0]
    v [:, 1] = [-0.5, 0]
    print("Initial velocity: ", v)
    collisions = initialisation(x, v, r, collisions, N, involvements)
    print(collisions)
    first = hq.heappop(collisions)
    x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, 1)
    print("Final velocity: ", v)

    #Collision parameter b = (r1+r2)/sqrt(2)
    print("Test 2: Collision parameter b = (r1 + r2)/sqrt(2)")
    b = (r[0] + r[1]) / np.sqrt(2)
    x[:, 0] = [0.25, 0.5 + b]
    x[:, 1] = [0.75, 0.5]
    v[:, 0] = [0.5, 0]
    v [:, 1] = [-0.5, 0]
    print("Initial velocity: ", v)
    v0 = np.array([[0.5, - 0.5], [0, 0]]) #Initial velocities
    collisions = initialisation(x, v, r, collisions, N, involvements)
    first = hq.heappop(collisions)
    x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, 1)
    cosine1 = np.dot(v0[:, 0], v[:, 0]) / (np.linalg.norm(v0[:, 0]) * np.linalg.norm(v[:, 0]))
    cosine2 = np.dot(v0[:, 1], v[:, 1]) / (np.linalg.norm(v0[:, 1]) * np.linalg.norm(v[:, 1]))
    print("Final velocity: ", v)
    print("Cosines to angles between incoming and outgoing particle velocities (expect 0):")
    print("Particle 1: ", cosine1)
    print("Particle 2: ", cosine2)

    #Two identical particles stop when they hit each other (xi=0)
    print("Test 3: Identical particles stop for xi = 0")
    xi = 0
    x[:, 0] = [0.25, 0.5]
    x[:, 1] = [0.75, 0.5]
    v[:, 0] = [0.5, 0]
    v [:, 1] = [-0.5, 0]
    print("Initial velocity: ", v)
    collisions = initialisation(x, v, r, collisions, N, involvements)
    first = hq.heappop(collisions)
    x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, 1)
    print("Final velocity: ", v)

def manyParticleTest(N, numberOfCollisions):
    v0 = 1
    theta = 2 * np.pi * np.random.random(N)
    v = np.array([v0 * np.cos(theta), v0 * np.sin(theta)]) #Particle velocities
    r = 1 / (4 * N) * np.ones(N) #Particle radii
    x = noOverlaps(r, N)
    m = np.random.random(N) #Particle masses
    collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
    involvements = np.zeros(N)
    xi = 1

    E0 = np.sum(1/2 * m * (v[0]**2 + v[1]**2))
    KE = [1] #Array of normalized kinetic energies
    times = np.linspace(0, numberOfCollisions, 10)

    collisions = initialisation(x, v, r, collisions, N, involvements)
    first = hq.heappop(collisions)

    for i in trange(len(times) - 1): #-1 is because we set the first element in the energy array beforehand
        x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions)
        E = np.sum(1/2 * m * (v[0]**2 + v[1]**2)) / E0 #Normalized kinetic energy
        KE.append(E)
    
    plt.plot(times, KE)
    plt.title("Energy Conservation in Gas of %d Particles" % N)
    plt.ylabel("E/E$_{0}$")
    plt.xlabel("Collisions")
    plt.show()



#oneParticleTests()

twoParticleTests()

#manyParticleTest(100, 1000)


# N = 1000
# r = 0.001 * np.ones(N) #Particle radii
# x = noOverlaps(r,N) #Particle positions; x[0][i] is the x-coord of particle i

# plt.figure(1)
# plt.xlim((0,1))
# plt.ylim((0,1))
# plt.scatter(x[0], x[1], s = 10000*np.amax(r))
# plt.title("Initial")
# v0 = 10
# theta = 2 * np.pi * np.random.random(N)
# v = np.array([v0 * np.cos(theta), v0 * np.sin(theta)]) #Particle velocities

# m = 0.1 * np.random.random(N) #Particle masses
# collisions = [] #Collision information: (collision time, particle index i, particle index j/wall, collision number)
# involvements = np.zeros(N)
# xi = 1

# collisions = initialisation(x, v, r, collisions, N, involvements)
# first = hq.heappop(collisions)
# x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, 1000)

# print("Max x:" ,np.amax(x[0]), "Max y:", np.amax(x[1]))
# plt.figure(2)
# plt.xlim((0,1))
# plt.ylim((0,1))
# plt.scatter(x[0], x[1], s = 10000*np.amax(r))
# plt.title("Final")
# plt.show()

#profile.print_stats()

          
# N = 1000
# r = 1 / (N//2) * np.ones(N) #Particle radii
# x = noOverlaps(r,N) #Particle positions; x[0][i] is the x-coord of particle i
# D = pdist(np.transpose(x))
# print("2 r = ", 2*r[0])
# print(np.amin(D), np.amin(D)>2*r[0])

# def noOverlaps(r, N):
#     x = np.empty((2, N))
#     x[:,0] = [0.5, 0.75]
#     for d in trange(1, 6): #Five domains to put particles into
#         x_ = np.empty((2, (N-1)//5))
#         x_[0, 1] = (0.2 * (d - 1) -  2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N])
#         x_[1, 1] = (0.1 * (d - 1) -  2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N])
#         particleList = [(x_[0, 1], x_[1, 1], r[1])]
#         for i in trange(2, (N-1)//5):
#             valid = False
#             while valid == False:
#                 p1 = ((0.2 * (d - 1) -  2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N])  , (0.1 * (d - 1) -  2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N]), r[i])
#                 j = 0
#                 p2 = particleList[j]
#                 dist = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) 
#                 while dist >= p1[2] + p2[2]:
#                     if j == len(particleList) - 1: #If we are at the final index, and the while is satisfied, the particle is valid
#                         valid = True
#                         particleList.append(p1)
#                         x_[:, i] = np.array([p1[0], p1[1]])
#                         break
#                     j += 1
#                     p2 = particleList[j]
#                     dist = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )
#         x[:, 1 + (d-1) * (N-1)//5:d * (N-1)//5] = x_
    #return x
    
    
# N = 2501
# r = 1 / np.sqrt(4 * np.pi * N) * np.ones(N) #Particle radii, ensures that the area taken up is 1/4
# r[0] = 5 * r[0] #Set projectile radius to 5 times that of the other particles
# tic = time.time()
# x = noOverlaps(r, N) #Particle positions; x[0][i] is the x-coord of particle i
# toc = time.time()
# print(toc - tic)
# plt.scatter(x[0], x[1])
# plt.show()

# def noOverlaps(r, N):
#     x = np.empty((2, N))
#     x[:,0] = np.random.uniform(0 + np.amax(r), 1 - np.amax(r), 2)
#     for i in trange(1, N):
#         valid = False
#         while valid == False:
#             #Need to pass how many particles have been placed to the function isValid
#             x_new = np.random.uniform(0 + np.amax(r), 1 - np.amax(r))
#             y_new = np.random.uniform(0 + np.amax(r), 1 - np.amax(r))
#             r_new = r[i]
#             valid = checkValidPos(x_new, y_new, r_new, x, r, i)
#         x[:, i] = np.array([x_new, y_new])
#     return x
    
# N = 5000
# r = 1 / np.sqrt(4 * np.pi * N) * np.ones(N)
# x = noOverlaps(r, N)
# D = pdist(np.transpose(x))
# print("2 r = ", 2*r[0])
# print(np.amin(D), np.amin(D)>2*r[0])

# plt.scatter(x[0], x[1], s = 1000 * r[0])
# plt.show()