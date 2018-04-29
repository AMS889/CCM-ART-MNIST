import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from scipy import io as spio
from torchvision import utils
from sklearn.preprocessing import MinMaxScaler

def get_random_subset_array(digit, npairs=20):
	"""
	Get randomly selected pairs of a single digit where it searches an array, y_test.
	Returns indices of images and the image tensors
	#### PARAMETERS ####
	digit: integer between 0 and 9
	npairs: default=20; number of pairs to return
	"""
    ix = np.where(y_test==int(digit))[0]
    np.random.shuffle(ix)
    ix1 = ix[:npairs]
    ix2 = ix[npairs:npairs*2]
    images1 = x_test[ix1]
    images2 = x_test[ix2]
    plt.figure(1,figsize=(npairs/5, npairs*2))
    plot_image_pairs(images1, images2)
    plt.show()
    plt.close()
    return ix1, ix2, images1, images2

def get_random_subset_tensor(digit, obj_type='array', npairs=20):
	"""
	Get randomly selected pairs of a single digit.
	Returns indices of images and the image tensors
	#### PARAMETERS ####
	digit: integer between 0 and 9
	obj_type: default='array'
		'array': search the numpy array of targets
		'tensor': search the test_loader targets
	npairs: default=20; number of pairs to return
	"""
    if obj_type.lower() == 'array':
    	ix = np.where(y_test==int(digit))[0]
	    np.random.shuffle(ix)
	    ix1 = ix[:npairs]
	    ix2 = ix[npairs:npairs*2]
	    images1 = x_test[ix1]
	    images2 = x_test[ix2]
	elif obj_type.lower() == 'tensor':
	    testiter = iter(test_loader)
	    images, target = testiter.next()
	    indices = np.flatnonzero(target.numpy() == digit_select)    
	    np.random.shuffle(indices)
	    ix1 = torch.LongTensor(indices[:npairs])
	    ix2 = torch.LongTensor(indices[npairs:npairs*2])
	    images1 = images[indx1]
	    images2 = images[indx2]
	else:
		raise ValueError('"obj_type" input invalid. Please use "array" or "tensor" only.')
    
    plt.figure(1,figsize=(4,40))
    plot_image_pairs(images1,images2)

    plt.figure(1,figsize=(npairs/5, npairs*2))
    plot_image_pairs(images1, images2)
    plt.show()
    plt.close()
    return ix1, ix2, images1, images2

def imshow(img):
	"""
	Display an image from the MNIST data set
	#### PARAMETERS ####
	img: torch tensor with two individual arrays inside for side-by-side comparison
	"""
    npimg = img.numpy()
    npscale = np.zeros(npimg.shape)
    for i,x in enumerate(npimg):
        npscale[i:,:,:] = 1 - scaler.transform(x)
    plt.imshow(np.transpose(npscale, (1, 2, 0)))
    plt.axis('off')
    
def plot_image_pairs(images1,images2,scores_net=[],scores_people=[], save=False, fname='digits.png'):
	"""
	Plot image pairs for comparison
	#### PARAMETERS ####
	images1: torch tensor
	images2: torch tensor to compare to impages1
	scores_net: default=[]; list of NN similarity scores
	scores_people: default=[]; list of human similarity scores
	save: default=False; binary indicator to save image file
	fname: default='digits.png'; file name for image file if save==True.
	"""
    try:
    	npairs = images1.shape[0]
    except:
    	images1.size()[0]
    	
    plt.figure(1,figsize=(npairs/5, npairs*2))
    assert images2.shape[0] == npairs
    for i in range(npairs):
        ax = plt.subplot(npairs, 1, i+1)
        ax.set_ylabel('Pair: {0:d}'.format(i), fontsize=4)
        try:
        	imshow(utils.make_grid([torch.from_numpy(images1[i],),torch.from_numpy(images2[i],)]))
        except:
        	imshow(utils.make_grid([images1[i], images2[i]])
        mytitle = 'Pair: {0:d}'.format(i)
        if len(scores_net)>0:
            mytitle += 'net %.2f, ' % scores_net[i] 
        if len(scores_people)>0:
            mytitle += 'human. %.2f' % scores_people[i]
        if mytitle:
            plt.title(mytitle)
    plt.tight_layout()
    if save:
        plt.savefig(fname, bbox_inches='tight', dpi=360)
    else:
        plt.show()
    plt.close()