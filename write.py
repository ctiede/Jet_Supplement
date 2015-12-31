import numpy as np
import h5py 


f = h5py.File('input.h5','r')
Nt = f['Grid']['Nr'].size-1
Nr = f['Grid']['Nr'][0:-1].flatten()

Nq = f['Data']['Cells'].shape[1]
Index = f['Grid']['Index']

theZones = [ [ np.zeros(Nq, dtype=np.float64) for i in range(Nr[j]) ] for j in range(Nt)]

t_jph = np.zeros(Nt+1, dtype=np.float64)
t_jph = f['Grid']['t_jph'][0:-1]
r_iph = [ np.zeros(Nr[j], dtype=np.float64) for j in range(Nt)]

for j in range(Nt):
    #TrackData = np.zeros( (Nr[j], (Nq+1)), dtype= np.float64)
    TrackData = f['Data']['Cells'][Index[j][0]:Index[j+1][0]]
    r_iph[j] = [ TrackData[i][Nq-1] for i in range(Nr[j]) ]
    theZones[j] = TrackData

g = h5py.File('output.h5','a')
try:
    del g['Data']
except:
    print "empty Data"

try:
    del g['Grid']
except:
    print "empty Data"
    
f.copy('Data',g)
f.copy('Grid',g)

Rc=0.5
Eset=1
Gamma=5./3
for j in range(Nt):
    for i in range(Nr[j]):
        if r_iph[j][i] < Rc:
            theZones[j][i][1] = theZones[j][i][1] + (Gamma-1)*Eset/(4./3*np.pi*Rc**3.0);

for j in range(Nt):
    g['Data']['Cells'][Index[j][0]:Index[j+1][0]] = theZones[j]

g.close()
f.close()

#h5py.Group.copy(f['Data'], g['Data'], name=None)
#h5py.Group.copy(f['Grid'], g['Grid'], name=None)


    

