import icosahedron
import predictingScalar
import torch 
import os

netpath = '/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-4636_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar'
#netpath = '/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-4969_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-64-64-64-7_glayers-1-64-64-64-1_gactivation0-relu_residual5dscalar'





subjects = os.listdir('/home/u2hussai/scratch/dtitraining/prediction_cut_pad/')

sub =subjects[0]

datapath= '/home/u2hussai/scratch/dtitraining/prediction_cut_pad/'+sub+'/6/'
print(datapath)
ico = icosahedron.icomesh(m=4)

predictor = predictingScalar.residual5dPredictorScalar(datapath,
                                           datapath + 'dtifit',
                                           datapath + 'dtifit',
                                           datapath,
                                           netpath,
                                           B=1,
                                           H=5,
                                           Nc=16,
                                           Ncore=100,
                                           core=ico.core_basis,
                                           core_inv=ico.core_basis_inv,
                                           zeros=ico.zeros,
                                           I=ico.I_internal,
                                           J=ico.J_internal)

if torch.cuda.is_available():
    predictor.net=predictor.net.cuda()
predictor.predict(datapath)




