import icosahedron
import predictingScalar
import torch 
import os
import sys

#netpath = '/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1-InternalPaddingON_Ntrain-3118_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar'
#netpath = '/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1_Ntrain-4969_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-64-64-64-7_glayers-1-64-64-64-1_gactivation0-relu_residual5dscalar'


#netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1-InternalPaddingON_reflectionfix_Ntrain-1564_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar'
#netpath='/home/u2hussai/scratch/dtitraining/networks2/bvec-dirs-6_type-V1-InternalPaddingON_reflectionfix_Ntrain-1564_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar'
netpath='/home/u2hussai/scratch/dtitraining/networks/bvec-dirs-6_type-V1-InternalPaddingON_reflectionfix_cornersNotZero_Ntrain-1564_Nepochs-20_patience-7_factor-0.5_lr-0.0001_batch_size-1_interp-inverse_distance_3dlayers-9-448-448-448_glayers-64-64-64-64-1_gactivation0-relu_residual5dscalar'



#subjects = os.listdir('/home/u2hussai/scratch/dtitraining/prediction_cut_pad/')
#subjects = os.listdir('/home/u2hussai/project/u2hussai/scratch_14Sept21/dtitraining/prediction_cut_pad/')
#subjects = os.listdir('/home/u2hussai/scratch/prediction_other/')

subjects=os.listdir('/home/u2hussai/project/u2hussai/prediction_other/')
for i in range(0,20):
    sub =subjects[i]


    out_dir='/home/u2hussai/scratch/network_predictions_5_subjects_right-padding_no-corner-zero_correct-theta-padding/'+sub+'/'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print('making prediction on '+out_dir)

        #datapath= '/home/u2hussai/scratch/dtitraining/prediction_cut_pad/'+sub+'/6/'
        #datapath= '/home/u2hussai/project/u2hussai/scratch_14Sept21/dtitraining/prediction_cut_pad/'+sub+'/6/'
       # datapath= '/home/u2hussai/scratch/prediction_other/'+sub+'/6/'
        datapath= '/home/u2hussai/project/u2hussai/prediction_other/'+sub+'/6/'
        print(datapath)
        ico = icosahedron.icomesh(m=4)

        predictor = predictingScalar.residual5dPredictorScalar(datapath + 'diffusion/',
                                                datapath + 'dtifit/dtifit',
                                                datapath + 'dtifit/dtifit',
                                                datapath + '/',
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
            predictor.net=predictor.net.cuda().eval()


        predictor.predict(out_dir)




