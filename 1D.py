# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:45:44 2019

@author: Administrator
"""
import matplotlib.pyplot as plt
from matplotlib import style
style.use('fivethirtyeight')
import numpy as np
from scipy.stats import norm
np.random.seed(0)
X = np.linspace(-15,15,num=20)

X1 = X*np.random.rand(len(X))+7 # Create data cluster 1
X2 = X*np.random.rand(len(X))-7 # Create data cluster 2
X_tot = np.stack((X1,X2)).flatten() # Combine the clusters to get the random datapoints from above
class GM1D:
    def __init__(self,X,iterations):
        self.iterations = iterations
        self.X = X
        self.mu = None
        self.pi = None
        self.var = None
  
    def run(self):
        
        """
        Instantiate the random mu, pi and var
        """
        self.mu = [-5,9]
        self.pi = [1/2,1/2]
        self.var = [5,2]
        
        """
        E-Step
        """
        
        for iter in range(self.iterations):
            print("Iteration No."+str(iter)+":")
            """Create the array r with dimensionality nxK"""
            r = np.zeros((len(X_tot),2))  
  
            """
            Probability for each datapoint x_i to belong to gaussian g 
            """
            for c,g,p in zip(range(2),[norm(loc=self.mu[0],scale=self.var[0]),
                                       norm(loc=self.mu[1],scale=self.var[1]),],self.pi):
                r[:,c] = p*g.pdf(X_tot) # Write the probability that x belongs to gaussian c in column c. 
                                     
            """
            Normalize the probabilities such that each row of r sums to 1 and weight it by mu_c == the fraction of points belonging to 
            cluster c
            """
            for i in range(len(r)):
                r[i] = r[i]/(np.sum(pi)*np.sum(r,axis=1)[i])
            """Plot the data"""
            fig = plt.figure(figsize=(5,5))
            ax0 = fig.add_subplot(111)
            for i in range(len(r)):
                ax0.scatter(self.X[i],0,c=np.array([r[i][0],r[i][1],r[i][0]]),s=100) 
            """Plot the gaussians"""
            for g,c in zip([norm(loc=self.mu[0],scale=self.var[0]).pdf(np.linspace(-20,20,num=40)),
                            norm(loc=self.mu[1],scale=self.var[1]).pdf(np.linspace(-20,20,num=40))],['magenta','lime']):
                ax0.plot(np.linspace(-20,20,num=40),g,c=c)
            
        
            """M-Step"""
    
            """calculate m_c"""
            m_c = []
            for c in range(len(r[0])):
                m = np.sum(r[:,c])
                m_c.append(m) # For each cluster c, calculate the m_c and add it to the list m_c
            """calculate pi_c"""
            for k in range(len(m_c)):
                self.pi[k] = (m_c[k]/np.sum(m_c)) # For each cluster c, calculate the fraction of points pi_c which belongs to cluster c
            """calculate mu_c"""
            self.mu = np.sum(self.X.reshape(len(self.X),1)*r,axis=0)/m_c
            print("mu: "+str(self.mu))
            """calculate var_c"""
            var_c = []
            for c in range(len(r[0])):
                var_c.append((1/m_c[c])*np.dot(((np.array(r[:,c]).reshape(40,1))*(self.X.reshape(len(self.X),1)-self.mu[c])).T,(self.X.reshape(len(self.X),1)-self.mu[c])))
            print("var： "+str(var_c))
            plt.show()
            
            
GM1D = GM1D(X_tot,17)
GM1D.run()


