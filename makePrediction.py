import predicting
from training import path_from_modelParams
import sys

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

print(sys.argv)

datapath=sys.argv[1]#"/home/u2hussai/scratch/dtitraining/downsample/sub-124220/"
dtipath=sys.argv[2]#"/home/u2hussai/projects/ctb-akhanf/ext-data/hcp1200/deriv/hcp1200_dtifit/results/sub-124220/dtifit"
netpath=sys.argv[3]
predpath=sys.argv[4]
outpath=sys.argv[5]



pred = predicting.predictor(datapath,dtipath,netpath)
pred.loadXpredict(predpath)
pred.loadNetwork()
pred.predict()
pred.savePredictions(outpath + 'V1_network.nii.gz')


