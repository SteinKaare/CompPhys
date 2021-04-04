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
    
@jit(nopython=True)
def getParticleParticleTimes(i, x, v, r, N):
    indices = np.arange(N)
    dx_x = x[0] - x[0, i]
    dx_y = x[1] - x[1, i]

    dv_x = v[0] - v[0, i]
    dv_y = v[1] - v[1, i]
    
    dvdx = dv_x * dx_x + dv_y * dx_y
    dvdv = dv_x**2 + dv_y**2
    dxdx = dx_x**2 + dx_y**2
    Rij = r + r[i]
    d = dvdx**2 - dvdv * (dxdx - Rij**2)
   
    #Boolean conditions
    A = dvdx < 0
    B = d > 0
    valid = A*B
    #Get values that satisfy A*B
    d = d[valid]
    indices = indices[valid]
    dvdx = dvdx[valid]
    dvdv = dvdv[valid]

    t = - (dvdx + np.sqrt(d)) / (dvdv)
    return t, indices

#Set up initialisation
def initialisation(x, v, r, collisions, N, involvements):
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
        times, indices = getParticleParticleTimes(i, x, v, r, N)
        for index, j in enumerate(indices):
            hq.heappush(collisions, (times[index], i, j, 0, 0))
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
    times, indices = getParticleParticleTimes(i, x, v, r, N)
    for index, j in enumerate(indices):
        hq.heappush(collisions, (times[index] + simTime, i, j, involvements[i], involvements[j]))
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
        times, indices = getParticleParticleTimes(p, x, v, r, N)
        for index, l in enumerate(indices):
            hq.heappush(collisions, (times[index] + simTime, p, l, involvements[p], involvements[l]))
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
    for k in trange(numberOfCollisions):
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
