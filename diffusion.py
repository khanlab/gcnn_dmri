from nibabel import load
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import numpy as np
from stripy import sTriangulation
from stripy.spherical import xyz2lonlat
from icosahedron import icomesh
from dihedral12 import xy2ind
from scipy.spatial import KDTree
import dipy


class dti():
    """
    Class for dti results
    """
    def __init__(self):
        self.FA=[]
        self.L1=[]
        self.L2=[]
        self.L3=[]
        self.V1=[]
        self.V2=[]
        self.V3=[]
        self.MD = []
        self.MO = []
        self.S0 = []
        self.mask= []

    def load(self,pathprefix):
        if pathprefix is None:
            raise ValueError("Please provide path including prefix for dti data, prefix=...")
        self.FA = load(pathprefix+"_FA.nii.gz")
        self.L1 = load(pathprefix + "_L1.nii.gz")
        self.L2 = load(pathprefix + "_L2.nii.gz")
        self.L3 = load(pathprefix + "_L3.nii.gz")
        self.V1 = load(pathprefix + "_V1.nii.gz")
        self.V2 = load(pathprefix + "_V2.nii.gz")
        self.V3 = load(pathprefix + "_V3.nii.gz")

class diffDownsample():
    """
    Class to reduce bvecs of diffusion data
    """
    def __init__(self,diff,ico,interp='inverse_distance'):
        self.diff = diff
        self.down_nii=[]
        self.ico=ico
        self.interp=interp
        self.down_interp_matrix=[]
        if self.interp=='inverse_distance': self.makeDownInterpMatrix() #makes the interpolation matrix

    def makeDownInterpMatrix(self):
        """
        Function to make a inverse distance interpolation matrix for the six (12)-direction mesh
        :return: populate self.down_interp_matrix
        """
        N_ico = len(self.ico.six_direction_mesh.lons)
        for bvec_mesh in self.diff.bvec_meshes:
            N_bvec = len(bvec_mesh.lons)
            interp_matrix = np.zeros([N_ico, N_bvec])
            dist, idx = bvec_mesh.nearest_vertices(self.ico.six_direction_mesh.lons, self.ico.six_direction_mesh.lats, k=10)
            weights = 1 / dist
            for row in range(0, N_ico):
                norm = sum(weights[row])
                interp_matrix[row, idx[row]] = weights[row] / norm
            self.down_interp_matrix.append(interp_matrix)

    def downSampleFromList(self,voxels,anti_treat='same'):
        interp=self.interp
        #i, j, k = np.where(self.diff.mask.get_fdata() == 1)
        #voxels = np.asarray([i, j, k]).T
        #sz=self.diff.vol.shape
        #sz.extend([7])
        ico_signal=[]
        S0=[]
        for pid,p in enumerate(voxels):
            S0_per_point = []
            ico_signal_per_shell=[]
            for sid in range(0,2):
                location=[]
                location.extend(p)
                location.append(self.diff.inds[sid])
                location=tuple(location)
                S=self.diff.vol.get_fdata()[location]
                if sid==0:
                    S0_per_point.append(S)
                    continue
                ico_lons = self.ico.six_direction_mesh.lons
                ico_lats = self.ico.six_direction_mesh.lats
                if interp == 'nearest': temp, err = self.diff.bvec_meshes[sid - 1].interpolate(ico_lons, ico_lats, order=0,zdata=S)
                if interp == 'linear': temp, err = self.diff.bvec_meshes[sid - 1].interpolate(ico_lons, ico_lats, order=1,zdata=S)
                if interp == 'cubic': temp, err = self.diff.bvec_meshes[sid - 1].interpolate(ico_lons, ico_lats, order=3,zdata=S)
                if interp == 'inverse_distance': temp = np.matmul(self.down_interp_matrix[sid - 1],S )  # inverse distance
                new_temp = np.zeros(len(ico_lats))
                for t in range(0,len(ico_lats),2):# add something to take average of antipodal signal or just use top
                    S1=temp[t]
                    S2=temp[t+1]
                    if anti_treat=='same':
                        new_temp[t]=S1
                        new_temp[t+1]=S1
                    else:
                        new_temp[t]=0.5*(S1+S2)
                        new_temp[t+1]=0.5*(S1+S2)
                ico_signal_per_shell.append(new_temp)
            ico_signal.append(ico_signal_per_shell)
            S0.append(S0_per_point)
        return S0,ico_signal



class diffVolume():
    """
    Class for diffusion volume data
    """
    def __init__(self):
        self.vol = []
        self.bvals = []
        self.bvecs = []
        self.bvecs_kd=[]
        self.bvals_sorted=[]
        self.bvecs_sorted=[]
        self.inds=[]
        self.gtab = []
        self.mask=[]
        self.bvec_meshes=[]
        self.ico_mesh=[]
        self.interpolation_matrices=[]


    def getVolume(self, folder=None):
        """
        Gets volume data
        :param filename: Path of volume file
        :return: Fills out attributes of the class
        """
        if folder is None:
            raise ValueError("Please provide diffusion folder path, folder=...")
        self.vol = load(folder+"/data.nii.gz")
        self.bvals, self.bvecs = read_bvals_bvecs(folder+"/bvals",folder+"/bvecs")
        self.gtab =gradient_table(self.bvals,self.bvecs)
        self.mask = load(folder+"/nodif_brain_mask.nii.gz")


    def shells(self):
        """
        This will sort bvals and bvecs by shells
        :return: Fills out self.b(val/vec)s_sorted and self.inds
        """
        tempbvals=[]
        tempbvals=np.round(self.gtab.bvals,-2)
        inds_sort=np.argsort(tempbvals)
        bvals_sorted=self.gtab.bvals[inds_sort]
        bvecs_sorted=self.gtab.bvecs[inds_sort]
        tempbvals=np.sort(tempbvals)
        gradbvals=np.gradient(tempbvals)
        inds_shell_cuts=np.where(gradbvals!=0)
        shell_cuts=[]
        for i in range(int(len(inds_shell_cuts[0]) / 2)):
            shell_cuts.append(inds_shell_cuts[0][i * 2])
        shell_cuts.insert(0,-1)
        shell_cuts.append(len(bvals_sorted))
        print(shell_cuts)
        print(bvals_sorted.shape)
        temp_bvals=[]
        temp_bvecs=[]
        temp_inds=[]
        for t in range(int(len(shell_cuts)-1)):
            print(shell_cuts[t]+1,shell_cuts[t + 1])
            temp_bvals.append(bvals_sorted[shell_cuts[t]+1:1+shell_cuts[t+1]])
            temp_bvecs.append(bvecs_sorted[shell_cuts[t]+1:1+shell_cuts[t+1]])
            temp_inds.append(inds_sort[shell_cuts[t]+1:1+shell_cuts[t+1]])
        self.bvals_sorted=temp_bvals
        self.bvecs_sorted=temp_bvecs
        self.inds=temp_inds
        self.inds=np.asarray(self.inds)

        #make the kdtree for each shell
        for bvecs in self.bvecs_sorted:
            self.bvecs_kd.append(KDTree(bvecs,10))

    def makeBvecMeshes(self):
        """
        Makes bvec meshes for each shell > 0. Note that each shell will have double the amount of bvec directions
        because of antipodal symmetrization.
        :return: Fills out self.bvec_meshes
        """
        n_shells=len(self.inds) #this includes the S_0 "shell"
        for shell in range(1,n_shells):
            x = self.bvecs_sorted[shell][:, 0]
            y = self.bvecs_sorted[shell][:, 1]
            z = self.bvecs_sorted[shell][:, 2]
            lons, lats = xyz2lonlat(x,y,z)

            #nlons, nlats = xyz2lonlat(-x,-y,-z)
            #lons = np.concatenate((plons,nlons),0)
            #lats = np.concatenate((plats, nlats), 0)
            self.bvec_meshes.append(sTriangulation(lons,lats,tree=True))

    def makeInverseDistInterpMatrix(self,ico_mesh):
        #this is with stripy
        N_ico=len(ico_mesh.lons)
        for bvec_mesh in self.bvec_meshes:
            N_bvec=len(bvec_mesh.lons)
            interp_matrix = np.zeros([N_ico, N_bvec])
            dist,idx=bvec_mesh.nearest_vertices(ico_mesh.lons,ico_mesh.lats,k=10)
            weights=1/dist
            for row in range(0,N_ico):
                norm=sum(weights[row])
                interp_matrix[row,idx[row]]=weights[row]/norm
            self.interpolation_matrices.append(interp_matrix)

        #this is just the usual stuff
        #make two matrices Aup and Adwn
        # N_ico = len(ico_mesh.x)
        # x = ico_mesh.x
        # y = ico_mesh.y
        # z = ico_mesh.z
        # xyz=np.array([x,y,z]).T
        # for shell,bvec_mesh in enumerate(self.bvec_meshes):
        #     N_bvec=len(bvec_mesh.x)
        #     interp_matrix=np.zeros([N_ico,N_bvec])
        #     dist,idx=self.bvecs_kd[shell+1].query(xyz,k=10)
        #     weights=1/(dist)
        #     for row in range(0,N_ico):
        #         norm=sum(weights[row])
        #         interp_matrix[row,idx[row]]=weights[row]/norm
        #     self.interpolation_matrices.append(interp_matrix)

    def inverseDistanceInterp(self,r,kd,signal):
        dist, inds = kd.query(r, 10)
        print("r",r)
        print("dist", dist)
        print("inds", inds)
        sigs = signal[inds]
        dist = 1 / dist
        norm = sum(dist)
        return sum(dist * sigs) / norm

    def makeFlat(self,p_list,ico_mesh,interp='inverse_distance'):
        """
        This function returns the diffusion signal at voxels in p_list as a flat array to be passed to the network
        :param p_list: List of voxels
        :param ico_mesh: Initiated (mesh made, etc.) instance of icomesh class
        :param interp: Type of interpolation used ('nearest','linear','cubic','inverse_distance')
        :return: Flat array
        """

        n_shells = len(self.inds) #this includes the S_0 "shell"
        ico_signal=[]
        flat=[]
        S0=[]
        for pid,p in enumerate(p_list): #have to cycle through all points in p_list
            #print(pid)
            ico_signal_per_shell=[]
            flat_per_shell=[]
            S0_per_point=[]
            for sid in range(0,n_shells): #go through each shell also
                location=[]
                location.extend(p)
                location.append(self.inds[sid])
                location=tuple(location)
                S=self.vol.get_fdata()[location]
                if sid==0:
                    S0_per_point.append(S)
                    continue
                Stwice=S#np.concatenate([S,S],0) #insteal of symmterizing the signal lets symmterize by average on
                # the icosahedron
                #Stwice=self.bvec_meshes[sid-1].smoothing(S,np.ones_like(S),1e-4,0.1,0.01)[0]
                ico_lons=ico_mesh.interpolation_mesh.lons
                ico_lats = ico_mesh.interpolation_mesh.lats
                #temp, err=self.bvec_meshes[sid-1].interpolate(ico_lons,ico_lats,order=1,zdata=Stwice/np.mean(S0_per_point))
                #stripy interp
                #print(S0)
                #print(np.mean(S0))
                if interp =='nearest': temp, err=self.bvec_meshes[sid-1].interpolate(ico_lons,ico_lats,order=0,zdata=Stwice/np.mean(S0_per_point))
                if interp =='linear': temp, err=self.bvec_meshes[sid-1].interpolate(ico_lons,ico_lats,order=1,zdata=Stwice/np.mean(S0_per_point))
                if interp =='cubic': temp, err=self.bvec_meshes[sid-1].interpolate(ico_lons,ico_lats,order=3,zdata=Stwice/np.mean(S0_per_point))
                if interp =='inverse_distance': temp=np.matmul(self.interpolation_matrices[sid-1],Stwice/np.mean(S0_per_point)) #inverse distance
                
                new_temp=np.zeros(len(ico_lats))
                for t in range(0,len(ico_lats)):
                    S1=temp[t]
                    S2=temp[int(ico_mesh.antipodals[t])]
                    new_temp[t]=(0.5*(S1+S2))
                ico_signal_per_shell.append(new_temp)
                #flat_per_shell.append(self.sphere_to_flat(ico_signal_per_shell[sid],ico_mesh))
                flat_per_shell.append(self.sphere_to_flat(new_temp, ico_mesh))
            #if ico_signal_per_shell is not None:
            ico_signal.append(ico_signal_per_shell)
            flat.append(flat_per_shell)
            S0.append(S0_per_point)

        return S0, flat, ico_signal

    def sphere_to_flat(self,ico_signal,ico_mesh):
        H=ico_mesh.m+1
        w=5*(H+1)
        h=H+1
        flat=np.zeros([h,w])
        top_faces=[[1,2],[5,6],[9,10],[13,14],[17,18]]
        for c in range(0,5):
            face=top_faces[c]
            for top_bottom in face:
                signal_inds = ico_mesh.interpolation_inds[top_bottom]
                signal=ico_signal[signal_inds]
                i=ico_mesh.i_list[top_bottom]
                j=ico_mesh.j_list[top_bottom]
                i=np.asarray(i).astype(int)
                j=np.asarray(c*h+j+1).astype(int)
                flat[i,j]=signal
        strip_xy = np.arange(0, H - 1)
        #print(strip)
        for c in range(0,5): #for padding
            #col=(c+1)%5*h+1
            #flat[0,c*h+strip]=flat[strip-1,col]
            flat[0,c*h+1]=ico_signal[0] #northpole

            c_left = c
            x_left = -1
            y_left = strip_xy
            i_left, j_left = xy2ind(H, c_left, x_left, y_left)
            # print(i_left,j_left)

            c_right = (c - 1) % 5
            x_right = H - 2 - strip_xy
            y_right = H - 2
            i_right, j_right = xy2ind(H, c_right, x_right, y_right)

            flat[i_left,j_left]=flat[i_right,j_right]



        return flat













