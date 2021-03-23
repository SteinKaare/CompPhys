import numpy as np
from scipy.spatial.distance import pdist
import os
import matplotlib.pyplot as plt
import heapq as hq
import line_profiler
from numba import jit
from tqdm import trange
import sys
profile = line_profiler.LineProfiler()

#Particle positions; x[0][i] is the x-coord of particle i; x[:,i] is [xi,yi]
#-1 is horizontal wall, -2 is vertical wall

@jit(nopython = True)
def getNextParticleTime(i, j, x, v, r):
    dx = x[:, j] - x[:, i] # [xj-xi, yj-yi]
    dv = v[:, j] - v[:, i] # [vxj-vxi, vyj-vyi]
    Rij = r[i] + r[j]
    ProdVX = np.dot(dv,dx)
    ProdVV = np.dot(dv, dv)
    d = ( ProdVX )**2 - ProdVV * (np.dot(dx, dx) - Rij**2)

    if ProdVX >= 0:
        t = np.inf
    elif d <= 0:
        t = np.inf
    else:
        t = - (ProdVX + np.sqrt(d)) / ProdVV
    return t

@jit(nopython = True)
def getAllParticleTimes(i, x, v, r, N):
    times = np.zeros(N)
    indices = np.arange(N)
    to_remove = np.array([-1]) #Seed the array
    for j in indices:
        t = getNextParticleTime(i, j, x, v, r)
        if j == i or t == np.inf: #Do not push these, either same particle or t = inf
            to_remove = np.append(to_remove, j)
            continue
        times[j] = t
    to_remove = np.delete(to_remove, 0) #Remove the seed
    indices = np.delete(indices, to_remove) #Remove the values we do not want to push
    return times, indices

@jit(nopython = True) 
def checkValidPos(x_new, y_new, r_new, x, r, placed): #Checks for overlaps between a particle and the rest
    for i in range(placed):
        dist = np.sqrt((x_new - x[0, i])**2 + (y_new - x[1, i])**2)
        if dist < r_new + r[i]:
            return False
    return True
        
#Ensure no overlaps between particles and walls
def noOverlaps(r, N):
    x = np.empty((2, N))
    x[:,0] = np.random.uniform(0 + np.amax(r), 1 - np.amax(r), 2)
    for i in trange(1, N):
        valid = False
        while valid == False:
            #Need to pass how many particles have been placed to the function isValid
            x_new = np.random.uniform(0 + np.amax(r), 1 - np.amax(r))
            y_new = np.random.uniform(0 + np.amax(r), 1 - np.amax(r))
            r_new = r[i]
            valid = checkValidPos(x_new, y_new, r_new, x, r, i)
        x[:, i] = np.array([x_new, y_new])
    return x
    

#Set up initialisation
def initialisation(x, v, r, collisions,N, involvements):
    #Calculate wall collision times
    for i in range(N):
        if v[0][i] > 0:
            t = (1 - r[i] - x[0][i]) / v[0][i]
            hq.heappush(collisions, (t, i, -2, 0) )
        elif v[0][i] < 0:
            t = (r[i]- x[0][i]) / v[0][i]
            hq.heappush(collisions, (t, i, -2, 0) )
        if v[1][i] > 0:
            t = (1 - r[i] - x[1][i]) / v[1][i]
            hq.heappush(collisions, (t, i, -1, 0) )
        elif v[1][i] < 0:
            t = (r[i]- x[1][i]) / v[1][i]
            hq.heappush(collisions, (t, i, -1, 0) )
    #Calculate particle collision times

    for i in range(N):
        for j in range(i+1,N):
            t = getNextParticleTime(i, j, x, v, r)
            if t == np.inf: continue #Do not push collisions that will not happen
            hq.heappush(collisions, (t, i, j, 0, 0))
    return collisions

#Update velocities following wall collisions
@jit(nopython = True)
def updateVelocitiesWall(i, v, collision, xi):
    #Update velocities
    if collision[2] == -2:
        v[0][i] = - xi * v[0][i]
        v[1][i] = xi * v[1][i]
    elif collision[2] == -1:
        v[0][i] = xi * v[0][i]
        v[1][i] = -xi * v[1][i]
    return v

#Update velocities following particle collisions
@jit(nopython = True)
def updateVelocitiesParticles(i, j, x, v, r, m, xi):
    #Update velocities
    dx = np.array([ x[0][j] - x[0][i], x[1][j]-x[1][i] ])
    dv = np.array([ v[0][j] - v[0][i], v[1][j]-v[1][i] ])

    v[:, i] = v[:, i] + ( (1 + xi) * m[j] / (m[i] + m[j]) * np.dot(dx, dv) / np.dot(dx, dx)) * dx
    v[:, j] = v[:, j] - ( (1 + xi) * m[i] / (m[i] + m[j]) * np.dot(dx, dv) / np.dot(dx, dx)) * dx

    return v

#Calculate next collision after a wall collision
@profile
def nextCollisionWall(i, x, v, r, N, collisions, involvements, simTime):
    #Next collision with wall
    if v[0][i] > 0:
        t = (1 - r[i] - x[0][i]) / v[0][i]
        hq.heappush(collisions, (t + simTime, i, -2, involvements[i]) )
    elif v[0][i] < 0:
        t = (r[i]- x[0][i]) / v[0][i]
        hq.heappush(collisions, (t + simTime, i, -2, involvements[i]) )
    if v[1][i] > 0:
        t = (1 - r[i] - x[1][i]) / v[1][i]
        hq.heappush(collisions, (t + simTime, i, -1, involvements[i]) )
    elif v[1][i] < 0:
        t = (r[i]- x[1][i]) / v[1][i]
        hq.heappush(collisions, (t + simTime, i, -1, involvements[i]) )
    #Next collision with particles:
    times, indices = getAllParticleTimes(i, x, v, r, N)
    for j in indices:
        hq.heappush(collisions, (times[j] + simTime, i, j, involvements[i], involvements[j]))
    return collisions

#Calculate next collision after a particle collision
@profile
def nextCollisionParticles(i, j, x, v, r, N, collisions, involvements, simTime):
    particles = (i,j)
    for p in particles:
        #Next collision with walls
        if v[0][p] > 0:
            t = (1 - r[p] - x[0][p]) / v[0][p]
            hq.heappush(collisions, (t + simTime, p, -2, involvements[p]) )
        elif v[0][p] < 0:
            t = (r[p]- x[0][p]) / v[0][p]
            hq.heappush(collisions, (t + simTime, p, -2, involvements[p]) )
        if v[1][p] > 0:
            t = (1 - r[p] - x[1][p]) / v[1][p]
            hq.heappush(collisions, (t + simTime, p, -1, involvements[p]) )
        elif v[1][p] < 0:
            t = (r[p]- x[1][p]) / v[1][p]
            hq.heappush(collisions, (t + simTime, p, -1, involvements[p]) )
        #Next collision with particles:
        times, indices = getAllParticleTimes(p, x, v, r, N)
        for l in indices:
            hq.heappush(collisions, (times[l] + simTime, p, l, involvements[p], involvements[l]))
    return collisions

#Ensure the next collisions is valid
def getValidCollision(collisions, involvements):
    valid = False
    collision = collisions[0]
    i = collision[1]
    while valid == False:
        if collision[3] != involvements[i]: #Check if the involvements of particle i matches the one at the collision time
            hq.heappop(collisions)
            if len(collisions) == 0: return 0  #If all particles have come to a stop, there are no further collisions
            collision = collisions[0]
            i = collision[1]
            continue
        j = collision[2]
        if j < 0: #Check if the collision is with a wall; if so, the collision is valid at this point
            valid = True
            continue
        if collision[4] != involvements[j]: #Check if the involvements of particle j matches the one at the collision time
            hq.heappop(collisions)
            if len(collisions) == 0: return 0  #If all particles have come to a stop, there are no further collisions
            collision = collisions[0]
            i = collision[1]
            continue
        valid = True
    return hq.heappop(collisions)

#Simulation loop
@profile
def loop(x, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions):
    collision = first
    i = collision[1]
    dt = collision[0]
    simTime = 0 #The accumulated simulated time
    for k in range(numberOfCollisions):
        involvements[i] += 1
        x = x + v * dt
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

    return x, v, collisions, involvements

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
    print("Test 3: [v0, v0] strikes all walls before returning to initial position")
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

    for i in range(len(times) - 1): #-1 is because we set the first element in the energy array beforehand
        x, v, collisions, involvements = loop(x, v, r, m, xi, N, first, collisions, involvements, numberOfCollisions)
        E = np.sum(1/2 * m * (v[0]**2 + v[1]**2)) / E0 #Normalized kinetic energy
        KE.append(E)
    
    plt.plot(times, KE)
    plt.title("Energy Conservation in Gas of %d Particles" % N)
    plt.ylabel("E/E$_{0}$")
    plt.xlabel("Collisions")
    plt.show()