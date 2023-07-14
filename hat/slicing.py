import os
import sys
import numpy as np 

ifile=sys.argv[1]
ofolder=sys.argv[2]
size_x=eval(sys.argv[3])#lr
size_y=eval(sys.argv[4])#lr
size_z=eval(sys.argv[5])#lr
scale=eval(sys.argv[6])
if not os.path.exists(ofolder):
    os.makedirs(ofolder)
arr=np.fromfile(ifile,dtype=np.float32).reshape((size_x,size_y,size_z))

for x in range(size_x):
    ofile="x_%d.dat" %(x*scale)
    opath=os.path.join(ofolder,ofile)

    arr[x,:,:].tofile(opath)

for y in range(size_y):
    ofile="y_%d.dat" %(y*scale)
    opath=os.path.join(ofolder,ofile)

    arr[:,y,:].tofile(opath)

for z in range(size_z):
    ofile="z_%d.dat" %(z*scale)
    opath=os.path.join(ofolder,ofile)

    arr[:,:,z].tofile(opath)


