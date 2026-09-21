import numpy as np
import matplotlib.pyplot as plt 
import sys
import muram as mio

# Reads a single muram snapshot and packs it into a npy, fits, pyana, or whatever you want at the moment

path = sys.argv[1]
n_iter = int(sys.argv[2])
type = sys.argv[10]

xmin = int(sys.argv[3])
xmax = int(sys.argv[4])
ymin = int(sys.argv[5])
ymax = int(sys.argv[6])
zmin = int(sys.argv[7])
zmax = int(sys.argv[8])
skip = int(sys.argv[9])

if (type=='muramsub'):

	snap=mio.MuramSubSnap(path,n_iter)

elif (type=='muram'):
	
	snap = mio.MuramSnap(path,n_iter)

else:
    print ("Uknown file type. Exiting...")
    exit()

T = snap.Temp.transpose(1,2,0)
	
print ("info::muram_binary_loader::the original dimensions are: ", T.shape)

NX, NY, NZ = T.shape
	
print ("info :: original dimensions are: ", NX, NY, NZ)

T = snap.Temp[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip].transpose(1,2,0)

NX, NY, NZ = T.shape
print ("info :: output dimensions are: ", NX, NY, NZ)

# We want T, p, vx, vy, vz, Bx, By, Bz

atmout = np.zeros([9,NX,NY,NZ])
Tc = np.copy(T)
	
p = snap.Pres[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip].transpose(1,2,0)
rho = snap.rho[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip].transpose(1,2,0)
vz = snap.vx[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip].transpose(1,2,0)
vx = snap.vy[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip].transpose(1,2,0)
vy = snap.vz[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip].transpose(1,2,0)
Bz = snap.Bx[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip].transpose(1,2,0) * np.sqrt(4.0*np.pi)
Bx = snap.By[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip].transpose(1,2,0) * np.sqrt(4.0*np.pi)
By = snap.Bz[zmin:zmax, xmin:xmax:skip, ymin:ymax:skip].transpose(1,2,0) * np.sqrt(4.0*np.pi)

atmout[0,:,:,:] = Tc
atmout[1,:,:,:] = p
atmout[2,:,:,:] = rho
atmout[3,:,:,:] = vx
atmout[4,:,:,:] = vy
atmout[5,:,:,:] = vz
atmout[6,:,:,:] = Bx
atmout[7,:,:,:] = By
atmout[8,:,:,:] = Bz

genname = sys.argv[11]

outputname = path + genname + '_' + str(n_iter) + '.npy'

atmout = atmout.astype(np.float32)

print("info::muram_to_cube::saving to: ", outputname)

np.save(outputname, atmout, allow_pickle=False)




