#import cProfile
import sys
import numpy as np
import h5py

def readSimple( h5fil, group , dset, data, dtype):
    #h5fil = h5py.h5f.open( filename )
    h5grp = h5py.h5g.open( h5fil, group )
    h5dst = h5py.h5d.open( h5grp, dset )
    #h5spc = h5dst.get_space()
    #dims = h5spc.get_simple_extent_dims()
    #rdata = np.zeros(dims, dtype=np.float64)
    h5dst.read(h5py.h5s.ALL, h5py.h5s.ALL, data, dtype)

    #del h5spc
    del h5dst
    del h5grp
    #del h5fil
    #return data


def getH5dims( h5fil, group, dset ):
    #h5fil = h5py.h5f.open( filename )
    h5grp = h5py.h5g.open( h5fil, group )
    h5dst = h5py.h5d.open( h5grp, dset )
    h5spc = h5dst.get_space()
    dims = h5spc.get_simple_extent_dims()

    del h5dst
    del h5grp
    del h5spc
    #del h5fil
    return dims


    
def readPatch( h5fil, group, dset, data, dtype, dim, start, loc_size,
               glo_size):
    #h5fil = h5py.h5f.open( filename )
    h5grp = h5py.h5g.open( h5fil, group )
    h5dst = h5py.h5d.open( h5grp, dset )
    h5spc = h5dst.get_space()

    mdims = np.zeros(dim, dtype=np.int32)
    fdims = np.zeros(dim, dtype=np.int32)
    
    fstart = np.zeros(dim, dtype=np.int32)
    fstride = np.zeros(dim, dtype=np.int32)
    fcount = np.zeros(dim, dtype=np.int32)
    fblock = np.zeros(dim, dtype=np.int32)
    
    for d in range(0, dim):
        mdims[d] = loc_size[d]
        fdims[d] = glo_size[d]
        
        fstart[d] = start[d]
        fstride[d] = 1
        fcount[d] = loc_size[d]
        fblock[d] = 1

    mspace = h5py.h5s.create_simple( tuple(mdims) ) # this is very important
    # fspace = h5py.h5s.create_simple( tuple(fdims) )
    # fspace.select_hyperslab(tuple(fstart), tuple(fcount),
    #                         tuple(fstride), tuple(fblock),h5py.h5s.SELECT_SET)
    # h5dst.read( mspace , fspace, data, dtype) 
    
    h5spc.select_hyperslab(tuple(fstart),
                           tuple(fcount),tuple(fstride),
                           tuple(fblock), h5py.h5s.SELECT_SET)
    h5dst.read( mspace , h5spc, data, dtype) 
    
    del mspace
    del h5dst
    del h5grp
    del h5spc
    #del h5fil
    
    del mdims
    del fdims
    del fstart
    del fstride
    del fcount
    del fblock

    
def readPatch_2( h5dst, h5spc, data, dtype, dim, start, loc_size, glo_size):
    
    mdims = np.zeros(dim, dtype=np.int32)
    fdims = np.zeros(dim, dtype=np.int32)
    
    fstart = np.zeros(dim, dtype=np.int32)
    fstride = np.zeros(dim, dtype=np.int32)
    fcount = np.zeros(dim, dtype=np.int32)
    fblock = np.zeros(dim, dtype=np.int32)
    
    for d in range(0, dim):
        mdims[d] = loc_size[d]
        fdims[d] = glo_size[d]
        
        fstart[d] = start[d]
        fstride[d] = 1
        fcount[d] = loc_size[d]
        fblock[d] = 1

    mspace = h5py.h5s.create_simple( tuple(mdims) ) # this is very important
    # fspace = h5py.h5s.create_simple( tuple(fdims) )
    # fspace.select_hyperslab(tuple(fstart), tuple(fcount),
    #                         tuple(fstride), tuple(fblock),h5py.h5s.SELECT_SET)
    # h5dst.read( mspace , fspace, data, dtype) 
    
    h5spc.select_hyperslab(tuple(fstart),
                           tuple(fcount),tuple(fstride),
                           tuple(fblock), h5py.h5s.SELECT_SET)
    h5dst.read( mspace , h5spc, data, dtype) 
    
    del mspace
    #del fspace
    # del h5dst
    # del h5grp
    # del h5spc
    # del h5fil
    
    del mdims
    del fdims
    del fstart
    del fstride
    del fcount
    del fblock
    
    #return data

def output(filename, theZones, t_jph, r_iph, Nt, Nr, Nq):
    f = open(filename, "w+")
    for j in range(Nt):
        for i in range(Nr[j]):
            f.write("%.12f\t%.12f\t"%(t_jph[j], r_iph[j][i]),)
            for k in range(Nq):
                f.write("%.12f\t"%(theZones[j][i][k]),)
        f.write("\n")
    f.close()
    
                


# class theZones_data:
#     def __init__(self, Nq):
#         self.data = np.zeros(Nq, dtype=np.float64)
#     def set(self, data):
#         self.data= data
        
# class theZones_i:
#     def __init__(self, Nr):
#         self.data = np.array(Nr, dtype=object)
#     def set(self, Nq):
#         self.data = 
    


def pyjet_prim(filename, Cell_Center=True):
    """filename -> Data This is the python version of vjet_python2.c retun Data array,
    Data[:,0] is the theta, Data[:,1] is radius, Data[:2] is density etc."""
        
    group1 = "Grid"
    group2 = "Data"
    # f = h5py.File(filename, 'r')
    # f.close()  # here, we can't use del f, cause even delete the
    # # reference, the file is still open. could not be accessed by
    h5fil = h5py.h5f.open( filename )
        
    dims = np.array([])
    
    dims = getH5dims( h5fil, group1, "T")
    t = np.zeros(dims, dtype=np.float64)
    readSimple( h5fil, group1, "T", t, h5py.h5t.NATIVE_DOUBLE)
    dims = getH5dims( h5fil, group1, "t_jph")

    Nt = dims[0] - 1 # note here t_jph[0] = 0 is artificial.  
    Nr = np.zeros(Nt, dtype=np.int32)
    t_jph = np.zeros(Nt+1, dtype=np.float64)
    Tindex = np.zeros(Nt, dtype=np.int32)

    print "t=%.2f, Nt = %d\n"%(t, Nt)
    readSimple( h5fil, group1, "t_jph", t_jph, h5py.h5t.NATIVE_DOUBLE )

    #    t_jph = t_jph[1:] # this line remove the first element(0), t_jph=np.delete(t_jph,0)
    start = np.array([0,0], dtype=np.int32)
    loc_size = np.array([Nt,1], dtype=np.int32)
    glo_size = np.array([Nt,1], dtype=np.int32)
    
    readPatch( h5fil, group1, "Nr" , Nr, h5py.h5t.NATIVE_INT32, 2 , start, loc_size, glo_size)
    readPatch( h5fil, group1, "Index", Tindex, h5py.h5t.NATIVE_INT32, 2, start,
                        loc_size, glo_size)
    
    dims = getH5dims( h5fil, group2, "Cells")
    Nc = dims[0]
    Nq = dims[1] - 1;
    print "Nc = %d Nt = %d Nq=%d \n"%(Nc, Nt, Nq)

    theZones =[ [ np.zeros(Nq, dtype=np.float64) for i in range(Nr[j])] for j in range(Nt) ]
    
    r_iph = [ np.zeros(Nr[j], dtype=np.float64) for j in range(Nt) ]
    

    print "Zones Allocated"
    loc_size[1] = Nq + 1
    glo_size[0] = Nc
    glo_size[1] = Nq + 1

    
    h5grp = h5py.h5g.open( h5fil, group2 )
    h5dst = h5py.h5d.open( h5grp, "Cells" )
    h5spc = h5dst.get_space()
    
    zeros= np.zeros
    array= np.array
    float64 = np.float64
    
    for j in range(Nt):
        loc_size[0] = Nr[j]
        start[0] = Tindex[j]
        TrackData = zeros( (Nr[j], (Nq+1)), dtype= float64)
        # readPatch(filename, group2, "Cells", TrackData,
        # h5py.h5t.NATIVE_DOUBLE, 2, start, loc_size, glo_size)

        readPatch_2(h5dst, h5spc, TrackData,
                    h5py.h5t.NATIVE_DOUBLE, 2, start, loc_size, glo_size)
        
        r_iph[j]= [ TrackData[i][Nq] for i in range(Nr[j])]
        theZones[j] = [ [ TrackData[i][q] for q in range(Nq)]
                       for i in range(Nr[j])]
        
        # for i in range(Nr[j]):
        #     r_iph[j][i] = TrackData[i*(Nq+1) + Nq]
        #     for q in range(Nq):
        #         theZones[j][i][q] = TrackData[i*(Nq+1) + q]
        
        del TrackData
        
        #output("pytemp2.dat", theZones, t_jph, r_iph, Nt, Nr, Nq)
    if Cell_Center :
        Data1 = np.array([ [ 0.5*(t_jph[j+1] + t_jph[j]),  0.5*r_iph[j][i] if i == 0 else 0.5*(r_iph[j][i] + r_iph[j][i-1])] for j in range(Nt) for i in range(Nr[j]) ])
    else :
        Data1 = np.array([ [t_jph[j+1], r_iph[j][i]] for j in range(Nt) for i in range(Nr[j]) ])

    Data2 = np.array([ [theZones[j][i][k] for k in range(Nq)]  for j in
              range(Nt) for i in range(Nr[j]) ])
    Data = np.concatenate((Data1, Data2), axis=1)
    theZones = array(theZones)
    del h5dst
    del h5spc
    del h5grp
    del h5fil
    return Data

def get_dV( xp, xm ):

    r = .5*( xp[0]+xm[0] )
    th = .5*(xp[1]+xm[1] )
    dr = xp[0] - xm[0]
    dth = xp[1] - xm[1]
    dph = 2*np.pi
    r2    = (xp[0]*xp[0]+xm[0]*xm[0]+xp[0]*xm[0])/3.
    sinth = np.sin(th)*( np.sin(.5*dth)/(.5*dth) );
    return r2*sinth*dr*dth*dph

RHO = 0; PPP=1; UU1=2; UU2=3; UU3=4;
DEN = 0; TAU=1; SS1=2; SS2=3; SS3=4;
NUM_C=4; NUM_Q=0;
GAMMA_LAW= 1.666666666666667

def prim2cons( prim, cons, r, dV):
    rho = prim[RHO];
    Pp  = prim[PPP];
    vr  = prim[UU1];
    vt  = prim[UU2];
    v2 = vr*vr + vt*vt;
    gam = GAMMA_LAW;
    rhoe = Pp/(gam-1.);
    cons[DEN] = rho*dV;
    cons[TAU] = (.5*rho*v2 + rhoe)*dV;
    cons[SS1] = rho*vr*dV;
    cons[SS2] = r*rho*vt*dV;
    
    for q in range(NUM_C,NUM_Q):
        cons[q] = cons[DEN]*prim[q];
            #//cons[q] = dV*prim[q];            //cons[q] = dV;

            
def prim2cons_3D( prim, cons, r, th, dV):
    rho = prim[RHO];
    Pp  = prim[PPP];
    vr  = prim[UU1];
    vt  = prim[UU2];
    vp  = prim[UU3];

    v2 = vr*vr + vt*vt + vp*vp;
    gam = GAMMA_LAW;
    rhoe = Pp/(gam-1.);
    cons[DEN] = rho*dV;
    cons[TAU] = (.5*rho*v2 + rhoe)*dV;
    cons[SS1] = rho*vr*dV;
    cons[SS2] = r*rho*vt*dV;
    cons[SS3] = r*rho*np.sin(th)*vp*dV;
    
    for q in range(NUM_C,NUM_Q):
        cons[q] = cons[DEN]*prim[q];
            #//cons[q] = dV*prim[q];            //cons[q] = dV;

    
    

def pyjet_cons(filename, Dim=2, Cell_Center=True):
    """filename -> Data This is the python version of vjet_python2.c retun Data array,
    Data[:,0] is the theta, Data[:,1] is radius, Data[:2] is density etc."""
        
    group1 = "Grid"
    group2 = "Data"
    # f = h5py.File(filename, 'r')
    # f.close()  # here, we can't use del f, cause even delete the
    # # reference, the file is still open. could not be accessed by
    h5fil = h5py.h5f.open( filename )
        
    dims = np.array([])
    
    dims = getH5dims( h5fil, group1, "T")
    t = np.zeros(dims, dtype=np.float64)
    readSimple( h5fil, group1, "T", t, h5py.h5t.NATIVE_DOUBLE)
    dims = getH5dims( h5fil, group1, "t_jph")

    Nt = dims[0] - 1 # note here t_jph[0] = 0 is artificial.  
    Nr = np.zeros(Nt, dtype=np.int32)
    t_jph = np.zeros(Nt+1, dtype=np.float64)
    Tindex = np.zeros(Nt, dtype=np.int32)

    print "t=%.2f, Nt = %d\n"%(t, Nt)
    readSimple( h5fil, group1, "t_jph", t_jph, h5py.h5t.NATIVE_DOUBLE )

    
    #t_jph = t_jph[1:] # this line remove the first element(0), t_jph=np.delete(t_jph,0)
    start = np.array([0,0], dtype=np.int32)
    loc_size = np.array([Nt,1], dtype=np.int32)
    glo_size = np.array([Nt,1], dtype=np.int32)
    
    readPatch( h5fil, group1, "Nr" , Nr, h5py.h5t.NATIVE_INT32, 2 , start, loc_size, glo_size)
    readPatch( h5fil, group1, "Index", Tindex, h5py.h5t.NATIVE_INT32, 2, start,
                        loc_size, glo_size)
    
    dims = getH5dims( h5fil, group2, "Cells")
    Nc = dims[0]
    Nq = dims[1] - 1;
    global NUM_Q
    NUM_Q = Nq
    print "Nc = %d Nt = %d Nq=%d \n"%(Nc, Nt, Nq)
    theZones =[ [ np.zeros(Nq, dtype=np.float64) for i in range(Nr[j])] for j in range(Nt) ]
    theZones_cons =[ [ np.zeros(Nq, dtype=np.float64) for i in range(Nr[j])] for j in range(Nt) ]
    
    r_iph = [ np.zeros(Nr[j], dtype=np.float64) for j in range(Nt) ]
    
    print "Zones Allocated"
    loc_size[1] = Nq + 1
    glo_size[0] = Nc
    glo_size[1] = Nq + 1

    
    h5grp = h5py.h5g.open( h5fil, group2 )
    h5dst = h5py.h5d.open( h5grp, "Cells" )
    h5spc = h5dst.get_space()
    
    zeros= np.zeros
    array= np.array
    float64 = np.float64
    
    for j in range(Nt):
        loc_size[0] = Nr[j]
        start[0] = Tindex[j]
        TrackData = zeros( (Nr[j], (Nq+1)), dtype= float64)
        # readPatch(filename, group2, "Cells", TrackData,
        # h5py.h5t.NATIVE_DOUBLE, 2, start, loc_size, glo_size)

        readPatch_2(h5dst, h5spc, TrackData,
                    h5py.h5t.NATIVE_DOUBLE, 2, start, loc_size, glo_size)
        
        r_iph[j]= [ TrackData[i][Nq] for i in range(Nr[j])]
        theZones[j] = [ [ TrackData[i][q] for q in range(Nq)]
                       for i in range(Nr[j])]
        
        for i in range(Nr[j]):
            rm = 0.0;
            if i>0 : rm = r_iph[j][i-1]
            rp = r_iph[j][i]
            xp = np.array([rp, t_jph[j+1]])
            xm = np.array([rm, t_jph[j]])
            # xm = np.array([rm, t_jph[j]])
            dV = get_dV(xp, xm)
            if Dim==2:
                r = (3./4.)*(rp*rp*rp*rp - rm*rm*rm*rm)/(rp*rp*rp - rm*rm*rm);
                prim2cons(theZones[j][i], theZones_cons[j][i], r, dV);
            elif Dim==3:
                global NUM_C
                NUM_C = 5;
                r = (3./4.)*(rp*rp*rp*rp - rm*rm*rm*rm)/(rp*rp*rp - rm*rm*rm);
                th = .5*(t_jph[j+1] + t_jph[j]) 
                prim2cons_3D(theZones[j][i], theZones_cons[j][i], r, th, dV);
            else:
                r = (2./3.)*(rp*rp*rp - rm*rm*rm)/(rp*rp - rm*rm);
                prim2cons(theZones[j][i], theZones_cons[j][i], r, dV);
        del TrackData

    if Cell_Center :
        Data1 = np.array([ [ 0.5*(t_jph[j+1] + t_jph[j]),  0.5*r_iph[j][i] if i == 0 else 0.5*(r_iph[j][i] + r_iph[j][i-1])] for j in range(Nt) for i in range(Nr[j]) ])
    else :
        Data1 = np.array([ [t_jph[j+1], r_iph[j][i]] for j in range(Nt) for i in range(Nr[j]) ])
        
        #output("pytemp2.dat", theZones, t_jph, r_iph, Nt, Nr, Nq)
    
    
    Data2 = np.array([ [theZones_cons[j][i][k] for k in range(Nq)]  for j in
              range(Nt) for i in range(Nr[j]) ])
    Data = np.concatenate((Data1, Data2), axis=1)
    theZones = array(theZones)
    del h5dst
    del h5spc
    del h5grp
    del h5fil
    return Data
    
def pyjet_time(filename):
    """filename -> Simulation Time
    """
    group1 = "Grid"
    group2 = "Data"
    # f = h5py.File(filename, 'r')
    # f.close()  # here, we can't use del f, cause even delete the
    # # reference, the file is still open. could not be accessed by
    h5fil = h5py.h5f.open( filename )
        
    dims = np.array([])
    
    dims = getH5dims( h5fil, group1, "T")
    t = np.zeros(dims, dtype=np.float64)
    readSimple( h5fil, group1, "T", t, h5py.h5t.NATIVE_DOUBLE)

    return t

if __name__== "__main__":
    #    cProfile.run('main()')
    try:
        filename = sys.argv[1]
    except:
        print "Please specify the input file.\n"; sys.exit(1)

    Data = pyjet_prim(filename)
     
