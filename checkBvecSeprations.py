import diffusion
import matplotlib.pyplot as plt


datapath="/home/uzair/PycharmProjects/unfoldFourier/data/101006/Diffusion/Diffusion"


diff=diffusion.diffVolume()
diff.getVolume(datapath)
diff.shells()

diff.downSampleFromList('./data','test')

# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.scatter(diff.bvecs_sorted[1][0:6,0],diff.bvecs_sorted[1][0:6,1],diff.bvecs_sorted[1][0:6,2])