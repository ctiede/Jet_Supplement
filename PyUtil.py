import numpy as np
import matplotlib.tri as tri

def triang(Radius_array, Theta_array):
    
    xpoints_arr = np.multiply(Radius_array, np.cos(Theta_array))
    ypoints_arr = np.multiply(Radius_array, np.sin(Theta_array))
    xpoints_append = np.hstack((xpoints_arr, - xpoints_arr))
    ypoints_append = np.hstack((ypoints_arr, ypoints_arr))
    triang = tri.Triangulation(xpoints_append, ypoints_append)
    return triang

class Unit:
    T = 961.665
    V = np.power(10,4.0)/np.sqrt(2)
    E = np.power(10,51.0)
    L = 6.8*np.power(10,6.)
    M = 2.0*np.power(10,30.0)
    Den = 1./157.216
    Pre = np.power(10,18.)/np.power(6.8,3)
    G_const =  0.000392578641938
            
class Unit2:
    T = 96.166522241370458
    V = np.power(10,9.0)/np.sqrt(2)
    E = np.power(10,51.0)
    L = 6.8*np.power(10.,10.)
    M = 2.0*np.power(10.,33.0)
    Den = 1./0.157216
    Pre = np.power(10.,21.)/np.power(6.8,3)
    G_const = 0.00392588235 
