import os
import sys
import numpy as np 

ifolder=sys.argv[1]
ofile=sys.argv[2]
size_x=eval(sys.argv[3])#hr
size_y=eval(sys.argv[4])#hrhr
size_z=eval(sys.argv[5])#hr

res=np.zeros((size_x,size_y,size_z),dtype=np.float32)
counts=np.zeros((size_x,size_y,size_z),dtype=np.float32)
filelist=[f for f in os.listdir(ifolder) if ".dat" in f]

for f in filelist:
    ipath=os.path.join(ifolder,f)
    arr=np.fromfile(ipath,dtype=np.float32)
    spl=f.split("_")
    xyz=spl[0]
    idx=eval(spl[1])
    '''
    if xyz=="x":
        arr=arr.reshape((size_y,size_z))
        res[idx,:,:]=arr
    elif xyz=="y":
        arr=arr.reshape((size_x,size_z))
        res[:,idx,:]=arr
    elif xyz=="z":
        arr=arr.reshape((size_x,size_y))
        res[:,:,idx]=arr
    '''

    if xyz=="x":
        arr=arr.reshape((size_y,size_z))
        
        res[idx,:,:]+=arr
        counts[idx,:,:]+=np.ones((size_y,size_z),dtype=np.float32)
    elif xyz=="y":
        arr=arr.reshape((size_x,size_z))
        res[:,idx,:]+=arr
        counts[:,idx,:]+=np.ones((size_x,size_z),dtype=np.float32)
    elif xyz=="z":
        arr=arr.reshape((size_x,size_y))
        res[:,:,idx]+=arr
        counts[:,:,idx]+=np.ones((size_x,size_y),dtype=np.float32)
res=np.divide(res,counts)

res.tofile(ofile)