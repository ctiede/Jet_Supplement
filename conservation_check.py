import matplotlib
import PyJet
import numpy as np
import matplotlib.pyplot as plt
import sys

Input = sys.argv[1]

#Cons = PyJet.pyjet_prim(Input)
Cons = PyJet.pyjet_cons(Input, Dim=3)

T = PyJet.pyjet_time(Input)

Den = np.array(Cons[:,2])   #2 for den, 3 for energy, 4 for vr 
Energy = np.array(Cons[:,3])
Rotation = np.array(Cons[:,6])
Radius = np.array(Cons[:,1])

print "Volumn: %e"%(np.power(Radius[-1],3))+ " Mass: %f "%(np.sum(Den))+"Energy: %.16f "%(np.sum(Energy))+\
    "Rotation Momentum: %f\n"%(np.sum(Rotation))

