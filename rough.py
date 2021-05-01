import icosahedron
from mayavi import mlab
import stripy

ico=icosahedron.icomesh()
ico.get_icomesh()
ico.vertices_to_matrix()
xyz=ico.getSixDirections()

x=[]
y=[]
z=[]
for vec in ico.vertices:
    x.append(vec[0])
    y.append(vec[1])
    z.append(vec[2])

mlab.points3d(x,y,z)
mlab.points3d(xyz[:,0],xyz[:,1],xyz[:,2],color=(1,0,0),scale_factor=0.23)
mlab.show()