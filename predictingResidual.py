import icosahedron
import predicting
import torch
import os

#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-125_Nepochs-50_patience-10_factor-0.5_lr-0.01_batch_size-1_interp-inverse_distance_glayers-8-8-8-1_gactivation0-relu_residual5d'
#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-600_Nepochs-50_patience-10_factor-0
# .5_lr-0.01_batch_size-1_interp-inverse_distance_glayers-16-16-16-16-16-1_gactivation0-relu_residual5d'
#datapath= '/home/u2hussai/scratch/dtitraining/prediction_cut_pad/sub-176845/6/'




#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-1444_Nepochs-30_patience-10_factor-0.5_lr-0.001_batch_size-1_interp-inverse_distance_glayers-32-32-32-32-32-1_gactivation0-relu_residual5d'
#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-1981_Nepochs-30_patience-3_factor-0.5_lr-0.001_batch_size-1_interp-inverse_distance_3dlayers-1-96-96_glayers-96-96-1_gactivation0-relu_residual5d'
#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-1981_Nepochs-30_patience-3_factor-0.5_lr-0.001_batch_size-1_interp-inverse_distance_3dlayers-1-64-64-64-64-32_glayers-32-32-32-32-1_gactivation0-relu_residual5d'
#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-1981_Nepochs-30_patience-3_factor-0.5_lr-0.001_batch_size-1_interp-inverse_distance_3dlayers-1-16-16-16-16-16-16-16_glayers-16-16-16-16-16-16-16-1_gactivation0-relu_residual5d'
netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-4396_Nepochs-20_patience-3_factor-0.5_lr-0.001_batch_size-1_interp-inverse_distance_3dlayers-1-64-64-64-64-32_glayers-32-32-32-32-1_gactivation0-relu_residual5d'



subjects = os.listdir('/home/u2hussai/scratch/dtitraining/prediction_cut_pad/')


for sub in subjects:
    datapath= '/home/u2hussai/scratch/dtitraining/prediction_cut_pad/'+sub+'/6/'
    print(datapath)
    ico = icosahedron.icomesh(m=4)

    predictor = predicting.residual5dPredictor(datapath,netpath,
                                            B=1,
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



