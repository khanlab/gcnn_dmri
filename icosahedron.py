import stripy
#from mayavi import mlab
import numpy as np
import geodesic
from anti_lib import Vec
from stripy.spherical import xyz2lonlat
import matplotlib.cm as cm
import matplotlib as mpl
from dipy.core.sphere import (sphere2cart, cart2sphere)
from dihedral12 import xy2ind
from dihedral12 import padding_basis

class icomesh:
    """
    Class to handle all things icosahedron
    """
    def __init__(self,m=5):
        self.faces=[] #this is the full icosahedron
        self.vertices=[] #this is the full icosahedron
        self.grid=[]
        self.m=m  #parameter for geodesic mesh
        self.H = self.m + 1
        self.n=0  #parameter for geodesic mesh
        self.repeats=1 #parameter for
        self.freq=self.repeats * (self.m * self.m + self.m * self.n + self.n * self.n)
        self.face_list=[]
        self.i_list=[]
        self.j_list=[]
        self.interpolation_inds=[]
        self.interpolation_mesh=[]
        self.antipodals=[]
        self.six_direction_mesh=[] #its actually 12 directions with antipodal points having the same signal
        self.X_in_grid=[]
        self.Y_in_grid=[]
        self.Z_in_grid=[]

        #run here usually used commands on initializtion, can be removed if needed
        self.get_icomesh()
        self.vertices_to_matrix()
        self.grid2xyz()

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
        self.antipodals=antipodals.astype(int)


    def grid2xyz(self):
        """
        Function that gives X,Y,Z coordinates on icosahedron flat grid
        """
        H=self.m+1
        w=5*(H+1)
        h=H+1
        self.X_in_grid = np.zeros([h,w])
        self.Y_in_grid = np.zeros([h,w])
        self.Z_in_grid = np.zeros([h,w])
        top_faces=[[1,2],[5,6],[9,10],[13,14],[17,18]]
        for c in range(0,5):
            print(c)
            face = top_faces[c]
            for top_bottom in face:
                i=self.i_list[top_bottom]
                j=self.j_list[top_bottom]
                vecs=self.face_list[top_bottom]
                x=[]
                y=[]
                z=[]
                for vec in vecs:
                    x.append(vec[0])
                    y.append(vec[1])
                    z.append(vec[2])
                i=np.asarray(i).astype(int)
                j=np.asarray(c*h+j+1).astype(int)
                self.X_in_grid[i,j]=x
                self.Y_in_grid[i,j]=y
                self.Z_in_grid[i,j]=z  #fill out northpole
                #print(self.X_in_grid[:,c*h:(c+1)*h])

        strip_xy = np.arange(0, H - 1)

        for c in range(0, 5):  # for padding
            # col=(c+1)%5*h+1
            # flat[0,c*h+strip]=flat[strip-1,col]
            self.X_in_grid[0, c * h + 1] = 0  # northpole
            self.Y_in_grid[0, c * h + 1] = 0  # northpole
            self.Z_in_grid[0, c * h + 1] = 1  # northpole

            c_left = c
            x_left = -1
            y_left = strip_xy
            i_left, j_left = xy2ind(H, c_left, x_left, y_left)
            # print(i_left,j_left)

            c_right = (c - 1) % 5
            x_right = H - 2 - strip_xy
            y_right = H - 2
            i_right, j_right = xy2ind(H, c_right, x_right, y_right)

            self.X_in_grid[i_left, j_left] = self.X_in_grid[i_right, j_right]
            self.Y_in_grid[i_left, j_left] = self.Y_in_grid[i_right, j_right]
            self.Z_in_grid[i_left, j_left] = self.Z_in_grid[i_right, j_right]

        I, J, T = padding_basis(H=H)


        self.X_in_grid = self.X_in_grid[I[0, :, :], J[0, :, :]]
        self.Y_in_grid = self.Y_in_grid[I[0, :, :], J[0, :, :]]
        self.Z_in_grid = self.Z_in_grid[I[0, :, :], J[0, :, :]]

    # def plot_icosohedron(self,maxface=22):
    #     """
    #     A function that plots the icosahedron with i,j labels
    #     :return: plot the icosahedron
    #     """
    #     colors=[[1,0,0],
    #             [0,1,0],
    #             [0,0,1],
    #             [1,1,0],
    #             [0,1,1]]
    #     for f in range(0,maxface):
    #         for p in range(0,len(self.face_list[f])):
    #             pnt=self.face_list[f][p]
    #             i = self.i_list[f][p]
    #             j = self.j_list[f][p]
    #             if np.isnan(i)==0 & np.isnan(i)==0:
    #                 string= "%d, %d" % (i,j)
    #                 mlab.text3d(pnt[0],pnt[1],pnt[2],string,scale=0.05)
    #                 mlab.points3d(pnt[0],pnt[1],pnt[2],scale_factor=0.05)

    def getSixDirections(self):
        """
        Returns six directions in the middle of each chart of the top half of the icosahedron
        :return:
        """

        def vec2np(r):
            return np.asarray([r[0],r[1],r[2]])

        def mid_vector(r1,r2):
            #r1=vec2np(r1)
            #r2=vec2np(r2)
            rmid=(r1+r2)/2
            return rmid/np.sqrt((rmid*rmid).sum())

        pairs = [[0, 2], [2, 11], [11, 9], [9, 6], [6, 0]]
        topdown=[[1,3],[1,4],[1,10],[1,8],[1,7]]


        six_directions = np.zeros([6, 3])
        six_directions[0,:]=[0,0,1]
        # six_directions[0,:]=[-0.416,0,0.910]
        # six_directions[0,:]=[+0.416,0,0.910]
        # six_directions[0,:]=[0.910,-0.416,0]
        # six_directions[0,:]=[0.910,0.416,0]
        # six_directions[0,:]=[0,0.910,0.416]
        # six_directions[0,:]=[0,0.910,-0.416]

        for id,pair in enumerate(pairs):
            # if id == 0:
            #     r1 = vec2np(self.vertices[pair[0]])
            #     r2 = vec2np(self.vertices[pair[1]])
            #     mid= mid_vector(r1,r2)
            #     six_directions[id+1,:]=mid_vector(vec2np( self.vertices[1]),mid)
            # if id ==1:
            #     r1 =vec2np( self.vertices[pair[0]])
            #     r2 =vec2np( self.vertices[pair[1]])
            #     six_directions[id+1,:]=mid_vector(r1,r2)
            # if id ==2:
            #     r1 = vec2np( self.vertices[pair[0]])
            #     r2 = vec2np( self.vertices[pair[1]])
            #     mid=mid_vector(r1,r2)
            #     six_directions[id+1,:]=mid_vector(mid,vec2np( self.vertices[ topdown[id][1]]))
            # if id ==3:
            #     six_directions[id+1,:]=vec2np( self.vertices[topdown[id][1]])
            # if id ==4:
            #     six_directions[id+1,:]=vec2np( self.vertices[pair[0]])

            if id % 2 ==0:
                r1 = vec2np(self.vertices[pair[0]])
                r2 = vec2np(self.vertices[pair[1]])
                six_directions[id+1,:]=mid_vector(r1,r2)
            else:
                r1 =vec2np( self.vertices[topdown[id][0]])
                r2 =vec2np( self.vertices[topdown[id][1]])
                six_directions[id+1,:]=mid_vector(r2,r2)

        lons=[]
        lats=[]
        for p in range(0,len(six_directions)):
            x=six_directions[p,0]
            y=six_directions[p,1]
            z=six_directions[p,2]
            lonc,latc=xyz2lonlat(x,y,z)
            lons.append(lonc)
            lats.append(latc)
            lonc, latc = xyz2lonlat(-x, -y, -z)
            lons.append(lonc)
            lats.append(latc)
        self.six_direction_mesh=stripy.sTriangulation(lons,lats,tree=True,permute=True)

        #return six_directions

        # lons=[]
        # lats=[]
        # for p in range(0,len(self.vertices)):
        #     x = self.vertices[p][0]
        #     y = self.vertices[p][1]
        #     z = self.vertices[p][2]
        #     lonsc, latsc = xyz2lonlat(x, y, z)
        #     lons.append(lonsc)
        #     lats.append(latsc)
        # self.coarse_ico=stripy.sTriangulation(lons,lats,tree=True,permute=True)
        # midpts=self.coarse_ico.face_midpoints()
        # x,y,z = stripy.spherical.lonlat2xyz(midpts[0],midpts[1])
        # six_directions=np.zeros([6,3])
        # six_directions[0,0]=0
        # six_directions[0, 1] = 0
        # six_directions[0, 2] = 1
        # a=1
        # for f in range(0,20,4):
        #     six_directions[a, 0] = x[f]
        #     six_directions[a, 1] = y[f]
        #     six_directions[a, 2] = z[f]
        #     a=a+1
        # return six_directions



