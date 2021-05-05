import icosahedron
#from mayavi import mlab
import stripy
import diffusion
from joblib import Parallel, delayed
import numpy as np
import time

datapath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/HCP_1200_T1w_Diffusion_FS/100408/T1w/Diffusion"
dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-100408/dtifit"

diff=diffusion.diffVolume()
diff.getVolume(datapath)
diff.shells()
diff.makeBvecMeshes()

ico=icosahedron.icomesh()
ico.get_icomesh()
ico.vertices_to_matrix()
ico.getSixDirections()

i, j, k = np.where(diff.mask.get_fdata() == 1)
voxels = np.asarray([i, j, k]).T

#compute time before/after "initialization"
start=time.time()
diffdown=diffusion.diffDownsample(diff,ico)
test=diffdown.downSampleFromList(voxels[0:10000])
end=time.time()
print(end-start)

start=time.time()
diffdown=diffusion.diffDownsample(diff,ico)
test=diffdown.downSampleFromList([voxels[0]])
test=diffdown.downSampleFromList(voxels[1:10000])
end=time.time()
print(end-start)




#xyz=ico.getSixDirections()
#
#x=[]
#y=[]
#z=[]
#for vec in ico.vertices:
#    x.append(vec[0])
#    y.append(vec[1])
#    z.append(vec[2])
#
#mlab.points3d(x,y,z)
#mlab.points3d(xyz[:,0],xyz[:,1],xyz[:,2],color=(1,0,0),scale_factor=0.23)
#mlab.show()
