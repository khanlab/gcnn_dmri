import icosahedron
import predicting

#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-125_Nepochs-50_patience-10_factor-0.5_lr-0.01_batch_size-1_interp-inverse_distance_glayers-8-8-8-1_gactivation0-relu_residual5d'
#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-600_Nepochs-50_patience-10_factor-0
# .5_lr-0.01_batch_size-1_interp-inverse_distance_glayers-16-16-16-16-16-1_gactivation0-relu_residual5d'
#datapath= '/home/u2hussai/scratch/dtitraining/prediction_cut_pad/sub-518746/6/'

netpath='./data/bvec-dirs-6_type-V1_Ntrain-200_Nepochs-50_patience-10_factor-0.5_lr-0.01_batch_size-1_interp-inverse_distance_glayers-16-16-16-16-1_gactivation0-relu_residual5d'
datapath= './data/downsample_cut_pad/sub-100206/6/'

ico = icosahedron.icomesh(m=4)

predictor = predicting.residual5dPredictor(datapath,netpath,multigpu=False,core=ico.core_basis,
                                           core_inv=ico.core_basis_inv,zeros=ico.zeros,I=ico.I_internal,J=ico.J_internal)
predictor.net=predictor.net.cuda()
predictor.predict('./data/')



