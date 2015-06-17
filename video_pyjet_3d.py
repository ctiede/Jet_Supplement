import matplotlib
matplotlib.use('Agg')
from pylab import *
import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import matplotlib.tri as tri
import sys
import matplotlib.gridspec as gridspec


import PyJet


Input = sys.argv[1]
Output = sys.argv[2]

contournum = 2; #density:2 pressure:3 vr:4 vt:5  vp: 6 X1: 7 X2: 8 X3:9
plotnum0 = 2;
plotnum1 = 3;
plotnum2 = 4;
plotnum3 = 6;

PI = 3.1415926536
Data = PyJet.pyjet_prim(Input)
Time_code = PyJet.pyjet_time(Input)

Radius_arr = np.array(Data[:,1])
Phi_arr    = np.array(Data[:,0]) + PI/2.0

xpoints_arr = np.multiply(Radius_arr, np.cos(Phi_arr))
ypoints_arr = np.multiply(Radius_arr, np.sin(Phi_arr))
xpoints_append = np.hstack((xpoints_arr, -xpoints_arr));
ypoints_append = np.hstack((ypoints_arr, ypoints_arr));

data_arr = np.array(Data[:, contournum])
plot_arr0 = np.array(Data[:, plotnum0])
plot_arr1 = np.array(Data[:, plotnum1])
plot_arr2 = np.array(Data[:, plotnum2])
plot_arr3 = plot_arr2/np.sqrt(plot_arr1/plot_arr0*5./3) # Mach number
plot_arr4 = np.sqrt(plot_arr1/plot_arr0*5./3)  # sound speed


data_append = np.hstack((data_arr, data_arr));

data_append = np.log10(data_append)

triang = tri.Triangulation(xpoints_append, ypoints_append)

fig= plt.figure(figsize=(6,10))
gs = gridspec.GridSpec(3, 1, height_ratios=[3,1,1])

ax=plt.subplot(gs[0])



plt.xlim(-2.5,2.5)
plt.ylim(-2.5,2.5)



plt.gca().set_aspect('equal')
v = np.linspace(1,18, 100, endpoint=True)

cax=plt.tricontourf(triang, data_append, 100,cmap=plt.cm.jet_r)
#plt.colorbar(cax, ticks=[4, 6, 8, 10, 12,14,16])
plt.colorbar()
plt.title("t=(%.2e)"%(Time_code))


Angles=np.unique(Phi_arr);

Angle2=Angles[64]
print "Angle1 %g"%((Angle2 - PI/2)/PI*180)
idx2=np.argwhere(np.fabs(Phi_arr - Angle2)<0.000001)
R2 = Radius_arr[idx2]
Phi2 = Phi_arr[idx2]
value2_0 = plot_arr0[idx2]  #density
value2_1 = plot_arr1[idx2]  #pressure
value2_2 = plot_arr2[idx2]  #V_r
value2_3 = plot_arr3[idx2]  #mach number
value2_4 = plot_arr4[idx2]  #sound speed

xpoints_2 = np.multiply(R2, np.cos(Phi2))
ypoints_2 = np.multiply(R2, np.sin(Phi2))

ax.plot(xpoints_2,ypoints_2,'c-')
ax.plot(-xpoints_2,ypoints_2,'c-')

ax=plt.subplot(gs[1])

ax.plot(R2,value2_0,color='red', label=r'$\rho$') 
ax.plot(R2,value2_1,color='blue', label=r'$P$') 

plt.legend()

ax=plt.subplot(gs[2])

ax.plot(R2,value2_4,color='magenta', label=r'$C_s$')
ax.plot(R2,value2_2,color='cyan', label=r'$V_r$')
ax.plot(R2,value2_3,color='red', label=r'$\mathbf{M}$')
plt.xlabel(r'$R$ Angle:%g'%((Angle2-PI/2)/PI*180))

plt.legend()
plt.savefig("%s"%Output,dpi=600) 
#plt.show()

