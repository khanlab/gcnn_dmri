import predicting
from training import path_from_modelParams

def dotLoss(output, target):

    a = output[:,0:3]
    ap = target[:,0:3]
    a=F.normalize(a,dim=-1)
    lossa = a*ap
    lossa=lossa.sum(dim=-1).abs()
    eps=1e-6
    lossa[(lossa-1).abs()<eps]=1.0
    
    lossa = torch.arccos(lossa).mean()
    return lossa

datapath="/home/u2hussai/scratch/dtitraining/downsample/sub-124220/"
dtipath="/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-124220/dtifit"


pred = predicting.predictor(datapath,dtipath,'./')
pred.loadXpredict('./X_predict.npy')
pred.loadNetwork('/home/u2hussai/scratch/dtitraining/networks/Ntrain-60000_Nepochs-200_patience-20_factor-0.5_lr-0.01_batch_size-16_interp-inverse_distance_glayers-1-4-8-16-32-64_gactivation0-relu_linlayers-2880-64-32-16-8-3_lactivation0-relu_')
pred.predict()



