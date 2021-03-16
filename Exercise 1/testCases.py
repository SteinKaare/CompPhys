from Functions import *
import time


# oneParticleTests()

# twoParticleTests()

# manyParticleTest(100, 1000)


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

def noOverlaps(r, N):
    x = np.empty((2, N))
    x[:,0] = [0.5, 0.75]
    for d in trange(1, 6): #Five domains to put particles into
        x_ = np.empty((2, (N-1)//5))
        x_[0, 1] = (0.2 * (d - 1) -  2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N])
        x_[1, 1] = (0.1 * (d - 1) -  2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N])
        particleList = [(x_[0, 1], x_[1, 1], r[1])]
        for i in trange(2, (N-1)//5):
            valid = False
            while valid == False:
                p1 = ((0.2 * (d - 1) -  2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N])  , (0.1 * (d - 1) -  2 * np.amax(r[1:N])) * np.random.random() + np.amax(r[1:N]), r[i])
                j = 0
                p2 = particleList[j]
                dist = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 ) 
                while dist >= p1[2] + p2[2]:
                    if j == len(particleList) - 1: #If we are at the final index, and the while is satisfied, the particle is valid
                        valid = True
                        particleList.append(p1)
                        x_[:, i] = np.array([p1[0], p1[1]])
                        break
                    j += 1
                    p2 = particleList[j]
                    dist = np.sqrt( (p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 )
        x[:, 1 + (d-1) * (N-1)//5:d * (N-1)//5] = x_
    return x
    
    
N = 2501
r = 1 / np.sqrt(4 * np.pi * N) * np.ones(N) #Particle radii, ensures that the area taken up is 1/4
r[0] = 5 * r[0] #Set projectile radius to 5 times that of the other particles
tic = time.time()
x = noOverlaps(r, N) #Particle positions; x[0][i] is the x-coord of particle i
toc = time.time()
print(toc - tic)
plt.scatter(x[0], x[1])
plt.show()
