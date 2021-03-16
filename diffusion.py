from nibabel import load
from dipy.io import read_bvals_bvecs
from dipy.core.gradients import gradient_table
import numpy as np
from stripy import sTriangulation
from stripy.spherical import xyz2lonlat

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


class diffVolume():
    """
    Class for diffusion volume data
    """
    def __init__(self):
        self.vol = []
        self.bvals = []
        self.bvecs = []
        self.bvals_sorted=[]
        self.bvecs_sorted=[]
        self.inds=[]
        self.gtab = []
        self.mask=[]
        self.bvec_meshes=[]
        self.ico_mesh=[]


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
            plons, plats = xyz2lonlat(x,y,z)
            nlons, nlats = xyz2lonlat(-x,-y,-z)
            lons = np.concatenate((plons,nlons),0)
            lats = np.concatenate((plats, plats), 0)
            self.bvec_meshes.append(sTriangulation(lons,lats,tree=True))

    def makeFlat(self,p_list,ico_mesh):
        """
        This function returns the diffusion signal at voxels in p_list as a flat array to be passed to the network
        :param p_list: List of voxels
        :param ico_mesh: Initiated (mesh made, etc.) instance of icomesh class
        :return: Flat array
        """

        n_shells = len(self.inds) #this includes the S_0 "shell"
        for pid,p in enumerate(p_list): #have to cycle through all points in p_list
            for sid in range(1,n_shells): #go through each shell also
                location=[]
                location.extend(p)
                location.append(self.inds[sid])
                location=tuple(location)
                S=self.vol.get_fdata[location]
                Stwice=np.concatenate([S,S],0)
                #TODO once the "interpolation mesh" is ready in the icomesh class, use that and bvec_meshes to put
                # data onto that mesh. From there use face_list,i_list and j_list and some systemic indexing to move
                # data to 2D x n_shells size array. From this point you need to pad overlapping indices and THEN pad
                # overlapping charts (this is different from former). At this stage the arrays should be ready for
                # training.
                #self.bvec_meshes[sid-1].interpolate(icomesh.)




diff=diffVolume()
diff.getVolume("/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion")
diff.shells()
diff.makeBvecMeshes()













