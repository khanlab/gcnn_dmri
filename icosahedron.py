import stripy
from mayavi import mlab
import numpy as np
import geodesic
from anti_lib import Vec
from stripy.spherical import xyz2lonlat
import matplotlib.cm as cm
import matplotlib as mpl


class icomesh:
    """
    Class to handle all things icosahedron
    """
    def __init__(self,m=4):
        self.faces=[] #this is the full icosahedron
        self.vertices=[] #this is the full icosahedron
        self.grid=[]
        self.m=m  #parameter for geodesic mesh
        self.n=0  #parameter for geodesic mesh
        self.repeats=1 #parameter for
        self.freq=self.repeats * (self.m * self.m + self.m * self.n + self.n * self.n)
        self.face_list=[]
        self.i_list=[]
        self.j_list=[]
        self.interpolation_inds=[]
        self.interpolation_mesh=[]
        self.antipodals=[]


    def get_icomesh(self):
        self.vertices.extend([Vec(0.894427,0.000000,0.447214),
            Vec(0.000000,0.000000,1.000000),
            Vec(0.276393,0.850651,0.447214),
            Vec(0.723607,0.525731,-0.447214),
            Vec(-0.276393,0.850651,-0.447214),
            Vec(-0.000000,0.000000,-1.000000),
            Vec(0.276393,-0.850651,0.447214),
            Vec(0.723607,-0.525731,-0.447214),
            Vec(-0.276393,-0.850651,-0.447214),
            Vec(-0.723607,-0.525731,0.447214),
            Vec(-0.894427,0.000000,-0.447214),
            Vec(-0.723607,0.525731,0.447214)])
        self.faces=[[2,0,1],[0,2,3],
          [4,3,2],[3,4,5],
          [11, 2, 1], [2, 11, 4],
          [10, 4, 11], [4, 10, 5],
          [9, 11, 1], [11, 9, 10],
          [8, 10, 9], [10, 8, 5],
          [6, 9, 1], [9, 6, 8],
          [7, 8, 6], [8, 7, 5],
          [0, 6, 1], [6, 0, 7],
          [3, 7, 0], [7, 3, 5]
           ]

        self.grid = geodesic.make_grid(self.freq, self.m, 0)

    def grid_to_ij_upper(self):
        N=self.m +1
        length=int(N*(N+1)/2)-N
        ii=  np.zeros(length)
        jj = np.zeros(length)
        l=0
        for j in range(0, N):
            for i in range(N-j-1,0,-1):
                ii[l]=i
                jj[l]=j
                l=l+1
        #return self.m-ii, jj #this subtraction is to reverse the rows
        return ii, jj

    def grid_to_ij_lower(self):
        N=self.m+1
        length = int(N * (N + 1) / 2)-N
        ii = np.zeros(length)
        jj = np.zeros(length)
        l = 0
        for j in range(N-1, 0,-1):
            for i in range(N - j,N):
                ii[l] = i
                jj[l] = j
                l = l + 1
        #return self.m-ii, jj #this subtraction is to reverse the rows
        return ii, jj

    def vertices_to_matrix(self):
        """
        This function is where we construct the mapping from the vertices in the top half of the icosahedron to 5
        square matrices. Notice that to avoid overlaps, the top rows of each matrix are not included. These will need to
        be padded with columns from neighbouring charts. Finally, keep note that the north and south poles are treated
        seperately and are outputted as the very first and last item in the list, respectively.
        :return: Three nested lists are returned, face_list, i_list, j_list. The list face_list, this is arranged as
        faces of the icosahedron, with the exception of the first and last entry which are the north and south poles
        respectively. Within each face is a list of the vertices with their coordinates in Vec format of anti_lib.
        The lists i_list and j_list are corresponding analogously structured lists that provide the i,j indices for
        the matrix mapping. Note that for vertices in the bottom half of the icosahedron these lists have a value of
        nan.
        """
        H=self.m+1
        face_list=[]
        iu,ju = self.grid_to_ij_upper()
        il, jl = self.grid_to_ij_lower()
        i_list=[]
        j_list=[]
        edges=[(0,0,1),
               (1,0,0),
               (0,2,1),
               (2,1,0)]
        face_list.append([Vec(0, 0, 1)]) #starts at north pole
        i_list.append([0])
        j_list.append([0])
        for f in range(0,20):
            upper_lower=(f)%4
            face=self.faces[f]
            points=np.flip(geodesic.grid_to_points(self.grid,self.freq,True,
                                                [self.vertices[face[i]] for i in range(3)],
                                                edges[upper_lower]))
            points=[p.unit() for p in points]
            face_list.append(points)
            if upper_lower ==0:
                i_list.append(iu)
                j_list.append(ju)
            elif upper_lower==1:
                i_list.append(il)
                j_list.append(jl)
            else:
                i_list.append([np.NaN for i in range(0,len(points))])
                j_list.append([np.NaN for i in range(0, len(points))])
        face_list.append([Vec(0, 0, -1)]) #ends at south pole
        #i_list.append([H])
        #j_list.append([H])
        i_list.append([np.NaN])
        j_list.append([np.NaN])
        self.face_list=face_list
        self.i_list = i_list
        self.j_list = j_list

        lons = []
        lats = []
        current_ind=0
        for f in face_list:
            face_inds=[]
            for p in f:
                lonsc,latsc=xyz2lonlat(p[0],p[1],p[2])
                lons.append(lonsc)
                lats.append(latsc)
                face_inds.append(current_ind)
                current_ind+=1
            self.interpolation_inds.append(face_inds)

        self.interpolation_mesh = stripy.sTriangulation(lons,lats,tree=True,permute=True)

        #we need to make list of the antipodal vertices to each vertex. this can be done brute force
        #   1) loop through all points, in x,y,z,
        #   2) find lat, long for -x,-y,-z
        #   3) find closest vertex and save the index, done
        antipodals=np.zeros(len(self.interpolation_mesh.x))
        for i in range(0,len(self.interpolation_mesh.x)):
            x = self.interpolation_mesh.x[i]
            y = self.interpolation_mesh.y[i]
            z = self.interpolation_mesh.z[i]
            lon,lat=xyz2lonlat(-x,-y,-z)
            dist, id= self.interpolation_mesh.nearest_vertices(lon,lat,1)
            antipodals[i]=int(id[0][0])
        self.antipodals=antipodals

    def plot_icosohedron(self,maxface=22):
        """
        A function that plots the icosahedron with i,j labels
        :return: plot the icosahedron
        """
        colors=[[1,0,0],
                [0,1,0],
                [0,0,1],
                [1,1,0],
                [0,1,1]]
        for f in range(0,maxface):
            for p in range(0,len(self.face_list[f])):
                pnt=self.face_list[f][p]
                i = self.i_list[f][p]
                j = self.j_list[f][p]
                if np.isnan(i)==0 & np.isnan(i)==0:
                    string= "%d, %d" % (i,j)
                    mlab.text3d(pnt[0],pnt[1],pnt[2],string,scale=0.05)
                    mlab.points3d(pnt[0],pnt[1],pnt[2],scale_factor=0.05)

