import math
import random
import numpy as np
import matplotlib.pyplot as plt
import pdb
import minVar
import Functions as F
# import SyntheticProblem as Construction
import MovieLensProblem as Construction
import pandas as pd
import time


Max = 100000
N = 100000



# M = Construction.M
# K = Construction.K
# D = Construction.D

l = Construction.l
Theta = Construction.Theta
c = Construction.c

''' c is 100 x 30 x 3, i.e. M x K x D, where we have for every client, 30 arms, 
    and every arm has a 3-dimensional feature vector which is x_{i,a} in the paper =: c[i, a].
'''

class Fed_PE_class:

    def __init__(
            self, 
            c, 
            Theta, 
            horizon,
            l,
            E = None,
            Y = None,
            T = None,
            mu1 = None,
            mu2 = None,
            messages = None,
            x_axis = None,
            y_axis = None,
            time = None,
            local_potential = None,
            var_matrix = None,
            pi = None,
            regret = None,
            n = None
    ) -> None:
        self.c = c
        self.Theta = Theta
        self.horizon = horizon
        self.l = l

        [self.m, self.k, self.d] = np.shape(c)
    
        # alpha
        self.alpha = 2 * np.log(self.m * self.k * horizon * 10 * 2)
        k_0 = 1
        while k_0 * self.d < 2 * np.log(self.k * horizon * 10) + self.d * np.log(k_0) + self.d:
            k_0 += 0.1
        self.alpha = min(self.alpha, 2 * np.log(self.k * horizon * 10) + self.d * np.log(k_0) + self.d)

        # Storage vectors and matrix
        self.E = np.zeros([self.m, self.k, self.d]) if E is None else E
        self.Y = np.zeros([self.m, self.k]) if Y is None else Y
        self.T = np.zeros([self.m, self.k]) if T is None else T
        self.mu1 = np.zeros([self.m, self.k]) if mu1 is None else mu1
        self.mu2 = np.zeros([self.m, self.k]) if mu2 is None else mu2
        self.messages = [np.zeros([self.k, self.d]) + Max, np.zeros([self.k, 2])] if messages is None else messages
        self.x_axis = [0] if x_axis is None else x_axis
        self.y_axis = [0] if y_axis is None else y_axis
        self.time = 0 if time is None else time
        self.local_potential = np.zeros([self.m, self.k]) + 1 if local_potential is None else local_potential
        self.var_matrix = 0 if var_matrix is None else var_matrix
        self.pi = np.zeros([self.m, self.k]) if pi is None else pi
        self.regret = 0 if regret is None else regret
        self.n = 0 if n is None else n

    def find_gap(self, c):
        Gap,deltamin,deltamax = F.FindGap(self.c, self.Theta)
        return Gap,deltamin,deltamax
    
    def initialization(self):
        self.local_information = np.zeros([self.m, self.k, self.d]) + Max
        for i in range(self.m):
            for a in range(self.k):
                self.Y[i,a] += F.PullArm(a, self.c[i,a], self.Theta, 1)
                self.T[i,1] += 1
                self.local_information[i,a] = self.c[i,a] * self.Y[i,a] / (self.c[i,a].dot(self.c[i,a])) # this is \hat{theta_{i, a}}^0 in the paper
                self.E[i,a] = self.local_information[i,a] / np.sqrt(self.local_information[i,a].dot(self.local_information[i,a]))
        A, R = F.PotentialSets(self.local_potential)
        ''' Broadcast will aggregate all the thetas that were pulled in the first phase.
            Then it'll send back the aggregated thetas to the clients in the form of messages. 
            Messages is just the "averaged"  thetas.
        '''
        self.messages, self.var_matrix = F.Broadcast(1, self.local_information, self.messages, self.pi+1, A, R)

    def run(self, initialization = True):
        
        Gap,deltamin,deltamax = self.find_gap(self.c)

        if initialization:
            self.initialization()
        
        for p in range(1, self.horizon):
            print('phase', p)
            self.local_information = np.zeros([self.m, self.k, self.d]) + Max
            A, R = F.PotentialSets(self.local_potential)
            pi0 = np.zeros([self.m, self.k])

            # Arm Elimination
            ''' Keep in mind that so far, we've assessed the best option for each client, which means we've M "potential" arms.
                We need to eliminate them to K arms.
            '''
            for i in range(self.m):
                B,C = F.Estimation(self.c, i, p, self.n, self.messages, self.var_matrix, self.Y, self.T, A, R)
                b = max( (v, b) for b, v in enumerate(B) )[1]
                pi0[i,b] = 1
                self.mu1[i,b] = B[b]*C[b]
                self.mu2[i,b] = C[b]
                for a in range(self.k):
                    if self.local_potential[i,a] == 1:
                        self.mu1[i,a] = B[a]*C[a]
                        self.mu2[i,a] = C[a]
                        if B[b] - B[a] > np.sqrt(self.alpha * self.mu2[i,b] / self.l) + np.sqrt(self.alpha * self.mu2[i,a] / self.l):
                            self.local_potential[i, a] = 0

                if self.local_potential[i].dot(self.local_potential[i]) == 0:
                    print('wrong elimination, all arms deleted.')
                    pdb.set_trace()

            # Server: Optimization Problem
            A, R = F.PotentialSets(self.local_potential)
            self.pi, iterations = minVar.OptimalExperiment(self.E, A, R, 0.1)

            # Clients: Exploration      
            for i in range(self.m):    
                for a in A[i]:
                    if self.local_potential[i,a] == 1:
                        phase_reward = 0
                        number = F.Int(F.f(p) * self.pi[i,a])
                        for pull_number in range(number):
                            instance_reward = F.PullArm(a, self.c[i,a], self.Theta)
                            phase_reward += instance_reward
                            self.Y[i,a] += instance_reward
                            self.T[i] += 1
                            self.regret += Gap[i,a]
                        if number > 0:
                            self.local_information[i,a] = self.c[i,a] * phase_reward / number / (self.c[i,a].dot(self.c[i,a]))
            
            # Server: Aggregation   
            self.messages, self.var_matrix = F.Broadcast(p+1, self.local_information, self.messages, self.pi, A, R)

            n=0    
            for a in range(self.k):
                n += self.T[0,a]
            self.x_axis.append(n)
            self.y_axis.append(self.regret/self.m)
        
        return self.x_axis, self.y_axis
    


def Fed_PE(c, Theta, horizon):
    print('Fed-PE begins: ')
    [m,k,d] = np.shape(c)
    
    # alpha
    alpha = 2 * np.log(m * k * horizon * 10 * 2)
    k_0 = 1
    while k_0 * d < 2 * np.log(k * horizon * 10) + d * np.log(k_0) + d:
        k_0 += 0.1
    alpha = min(alpha, 2 * np.log(k * horizon * 10) + d * np.log(k_0) + d)
    
    # Gap
    Gap,deltamin,deltamax = F.FindGap(c, Theta)
    
    # Storage vectors and matrix
    E = np.zeros([m, k, d])
    Y = np.zeros([m, k])
    T = np.zeros([m, k])
    mu1 = np.zeros([m, k])
    mu2 = np.zeros([m, k])
    messages = [np.zeros([k, d]) + Max, np.zeros([k, 2])]
    x_axis = [0]
    y_axis = [0]
    time = 0
    local_potential = np.zeros([m, k]) + 1
    var_matrix = 0
    pi = np.zeros([m, k])
    regret = 0
    n = 0
    
    # Initialization
    ''' For every client, pull each arm once.
    '''
    local_information = np.zeros([m, k, d]) + Max
    for i in range(m):
        for a in range(k):
            Y[i,a] += F.PullArm(a, c[i,a], Theta, 1)
            T[i,1] += 1
            local_information[i,a] = c[i,a] * Y[i,a] / (c[i,a].dot(c[i,a])) # this is \hat{theta_{i, a}}^0 in the paper
            E[i,a] = local_information[i,a] / np.sqrt(local_information[i,a].dot(local_information[i,a]))
    A, R = F.PotentialSets(local_potential)
    ''' Broadcast will aggregate all the thetas that were pulled in the first phase.
        Then it'll send back the aggregated thetas to the clients in the form of messages. 
        Messages is just the "averaged"  thetas.
    '''
    messages, var_matrix = F.Broadcast(1, local_information, messages, pi+1, A, R)

    for p in range(1, horizon):
        print('phase', p)
        local_information = np.zeros([m, k, d]) + Max
        A, R = F.PotentialSets(local_potential)
        pi0 = np.zeros([m, k])
        
        # Arm Elimination
        ''' Keep in mind that so far, we've assessed the best option for each client, which means we've M "potential" arms.
            We need to eliminate them to K arms.
        '''
        for i in range(m):
            B,C = F.Estimation(c, i, p, n, messages, var_matrix, Y, T, A, R)
            b = max( (v, b) for b, v in enumerate(B) )[1]
            pi0[i,b] = 1
            mu1[i,b] = B[b]*C[b]
            mu2[i,b] = C[b]
            for a in range(k):
                if local_potential[i,a] == 1:
                    mu1[i,a] = B[a]*C[a]
                    mu2[i,a] = C[a]
                    if B[b] - B[a] > np.sqrt(alpha * mu2[i,b] / l) + np.sqrt(alpha * mu2[i,a] / l):
                        local_potential[i, a] = 0

            if local_potential[i].dot(local_potential[i]) == 0:
                print('wrong elimination, all arms deleted.')
                pdb.set_trace()
        
        # Server: Optimization Problem
        A, R = F.PotentialSets(local_potential)
        pi, iterations = minVar.OptimalExperiment(E, A, R, 0.1)

        # Clients: Exploration      
        for i in range(m):    
            for a in A[i]:
                if local_potential[i,a] == 1:
                    phase_reward = 0
                    number = F.Int(F.f(p) * pi[i,a])
                    for pull_number in range(number):
                        instance_reward = F.PullArm(a, c[i,a], Theta)
                        phase_reward += instance_reward
                        Y[i,a] += instance_reward
                        T[i,a] += 1
                        regret += Gap[i,a]
                    if number > 0:
                        local_information[i,a] = c[i,a] * phase_reward / number / (c[i,a].dot(c[i,a]))
        
        # Server: Aggregation   
        messages, var_matrix = F.Broadcast(p+1, local_information, messages, pi, A, R)
         
        n=0    
        for a in range(k):
            n += T[0,a]
        x_axis.append(n)
        y_axis.append(regret/m)
    '''
    print(Gap,deltamin,deltamax)
    print('whether delete optimal arm: ',np.trace(local_potential.dot(Gap.transpose())))
    print('whether delete optimal arm: ',np.trace(local_potential.dot(local_potential.transpose())))
    print('regret per client: ', regret/m)
    '''
    return x_axis, y_axis

horizon = 17
n = 1
S1 = np.zeros(horizon)
S2 = np.zeros(horizon)
S3 = np.zeros(horizon)
S4 = np.zeros(horizon)
S5 = np.zeros(horizon)
S6 = np.zeros(horizon)
S7 = np.zeros(horizon)
S8 = np.zeros(horizon)
S9 = np.zeros(horizon)
for i in range(n):
    # first 100 clients
    x_axis, S = Fed_PE(c[:50], Theta, horizon) 
    S1 = S1 + np.array(S)
    # first 200 clients
    x_axis_200, S_2 = Fed_PE(c[:100], Theta, horizon) 
    S2 = S2 + np.array(S_2)
    # first 100 clients with Fed_PE_class
    fedpe = Fed_PE_class(c[:50], Theta, horizon, l)
    _ , S_3 = fedpe.run()
    S3 = S3 + np.array(S_3[-horizon:])
    # second 100 clients with Fed_PE_class, but continued
    fedpe.c = c[50:100]
    _ , S_4 = fedpe.run(initialization=False)
    S4 = S4 + np.array(S_4)[-horizon:]
    # third 100 clients with Fed_PE_class, but continued
    fedpe.c = c[100:150]
    _ , S_7 = fedpe.run(initialization=False)
    S7 = S7 + np.array(S_7)[-horizon:]
    # second 100 clients with Fed_PE_class, but anew
    _ , S_5 = fedpe.run(initialization=True)
    S5 = S5 + np.array(S_5[-horizon:])
    # third 100 clients with Fed_PE_class, but anew
    fedpe.c = c[100:150]
    _ , S_8 = fedpe.run(initialization=True)
    S8 = S8 + np.array(S_8)[-horizon:]
    # all 200 clients with Fed_PE_class
    fedpe_2 = Fed_PE_class(c[:100], Theta, horizon, l)
    _ , S_6 = fedpe_2.run()
    S6 = S6 + np.array(S_6[-horizon:])
    # all 300 clients with Fed_PE_class
    fedpe_3 = Fed_PE_class(c[:150], Theta, horizon, l)
    _ , S_9 = fedpe_3.run()
    S9 = S9 + np.array(S_9[-horizon:])

S1 = S1/(n + 0.0)
S2 = S2/(n + 0.0)
S3 = S3/(n + 0.0)
S4 = S4/(n + 0.0)
S5 = S5/(n + 0.0)
S6 = S6/(n + 0.0)
S7 = S7/(n + 0.0)
S8 = S8/(n + 0.0)
S9 = S9/(n + 0.0)

plt.plot(np.array(x_axis), S1, '-.', label = 'Fed-PE 0-50')
plt.plot(np.array(x_axis), S2, '-', label = 'Fed-PE 0-100')
plt.plot(np.array(x_axis), S3, '-', label = 'Fed-PE-class 0-50')
plt.plot(np.array(x_axis), S4, '-.', label = 'Fed-PE-class 50-100 continued')
plt.plot(np.array(x_axis), S5, '-', label = 'Fed-PE-class 50-100 anew')
plt.plot(np.array(x_axis), S6, '-.', label = 'Fed-PE-class 0-100')
plt.plot(np.array(x_axis), S7, '-', label = 'Fed-PE-class 100-150 continued')
plt.plot(np.array(x_axis), S8, '-.', label = 'Fed-PE-class 100-150 anew')
plt.plot(np.array(x_axis), S9, ':', label = 'Fed-PE-class 0-150')





plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
plt.ylabel('Regret: R(T)')
plt.xlabel('Time: T')
plt.tight_layout()
plt.grid(ls='--')
plt.show()  
plt.savefig('Fed-PE.png')
