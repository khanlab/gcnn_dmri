import icosahedron
import os
import re
import nibabel as nib



class resPred:
    def __init__(self,path,net):
        self.path=path
        self.net=net

    def load(self,chunk_size=9):
        files = os.listdir(self.path + 'data_chunks/')

        X = []  # input
        Y = []  # output
        S0 = []

        file=files[0]
        midstring=file[7:34]
        parse = re.split(r'[-_.]', file)

        i_min= int(parse[3])
        i_max = int(parse[4])
        j_min = int(parse[6])
        j_max = int(parse[7])
        k_min = int(parse[9])
        k_max = int(parse[10])

        chnk_ind=0
        for i in range(i_min,i_max,chunk_size):
            for j in range(j_min,j_max,chunk_size):
                for k in range(k_min,k_max,chunk_size):
                    tempfile = [file for file in files if file.startswith('data_flat' + midstring + str(chnk_ind) +
                                                                          '_')]
                    data=nib.load(self.path+'data_chunks/'+tempfile[0]).get_fdata()
                    tempfile = [file for file in files if file.startswith('data_S0' + midstring + str(chnk_ind) +
                                                                          '_')]
                    S0=nib.load(self.path + 'data_chunks/' + tempfile[0]).get_fdata()




rp = resPred('./data/6/',1)
rp.load()
