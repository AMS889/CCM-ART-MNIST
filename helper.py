import matplotlib.pyplot as plt
import torch
import numpy as np
import random
from scipy import io as spio
from scipy.spatial.distance import cosine
from torchvision import utils
from torch.autograd import Variable

def get_random_subset(test_loader_ordered, use_cuda, digit, npairs=20):
    """
    Get randomly selected pairs of a single digit.
    Returns indices of images and the image tensors
    #### PARAMETERS ####
    test_loader_ordered: unshuffled data_loader test set
    digit: integer between 0 and 9
    npairs: default=20; number of pairs to return
    """
    # digit_select: which digit do we want to get images for?
    testiter = iter(test_loader_ordered)
    images, target = testiter.next()
    images = rotate(images, cuda=use_cuda)
    indices = np.flatnonzero(target.cpu().numpy() == digit)
    np.random.shuffle(indices)
    ix1 = torch.LongTensor(indices[:npairs])
    ix2 = torch.LongTensor(indices[npairs:npairs*2])
    im1 = images[ix1]
    im2 = images[ix2]
    plt.figure(1,figsize=(4,40))
    plot_image_pairs(im1,im2)
    return ix1, ix2, im1, im2

def get_aligned_images(data_loader):
    """
    Returns the rotated images and targets in a torch dataset object for EMNIST
    """
    dataiter = iter(data_loader)
    images, target = dataiter.next()
    images = rotate(images)
    return images, targets

def rotate(data, cuda=False):
    """
    Rotates all the data in a batch of a torch dataset object
    """
    data2 = data.cpu()
    for ix, d in enumerate(data2):
        data2[ix] = torch.FloatTensor(d.numpy().flatten(order='F').reshape(1,1,28,28))
    data = data2.cuda() if cuda else data2.cpu()
    return data

def imshow(img, data_loader=False):
    """
    Display an image from the MNIST data set
    #### PARAMETERS ####
    img: torch tensor with two individual arrays inside for side-by-side comparison
    """
    img = 1 - (img.numpy() * 0.3081 + 0.1307) # invert image pre-processing
    plt.imshow(np.transpose(img, (1, 2, 0)))
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
    npairs = images1.size()[0]
    assert images2.size()[0] == npairs
    if npairs > 20:
        plt.figure(1,figsize=(npairs/5, npairs*2))
    for i in range(npairs):
        ax = plt.subplot(npairs, 1, i+1)
        imshow(utils.make_grid([images1[i], images2[i]]))
        mytitle = 'Pair: {0:d}'.format(i)
        if len(scores_net)>0:
            mytitle += ' net %.2f, ' % scores_net[i]
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

def test_viz(model, data_loader, nshow=10):
    # grab a random subset of the data
    dataiter = iter(data_loader)
    images, target = dataiter.next()
    images = rotate(images)
    perm = np.random.permutation(images.size()[0])
    sel = torch.LongTensor(perm[:nshow])
    images = images[sel]
    data = Variable(images, volatile=True).cpu()

    # get predictions from the network
    output, h_fc1, h_conv2, h_conv1 = model(data)
    pred = output.data.max(1, keepdim=True)[1]
    pred = pred.numpy().flatten()

    # plot predictions along with the images
    for i in range(nshow):
        ax = plt.subplot(1, nshow, i+1)
        imshow(utils.make_grid(images[i]))
        plt.title(str(pred[i]))

def normalize(v):
    """
    Normalize v to [0,1]
    """
    v = v - v.min()
    v = v / v.max()
    return v

def get_similarity_scores(images1, images2, model, layer='fc'):
    """
    Returns the similarity scores from a given model Layer
    #### PARAMETERS ####
    images1: list of images
    images2: list of images
    model: trained CNN
    layer: default='fc': network layer of interest
        'fc': fully connected
        'conv1': first convolutional
        'conv2': second convolutional
        'conv3': second convolutional
        'conv4': second convolutional
    """
    N = images1.size()[0] # number of pairs
    assert N == images2.size()[0]
    with torch.no_grad():
        images1 = Variable(images1)
        images2 = Variable(images2)
    outputs1 = model(images1)
    outputs2 = model(images2)

    # grab the tensors from the appropriate layer
    if layer=='fc':
        T1 = outputs1[-1]
        T2 = outputs2[-1]
    elif layer=='conv1':
        T1 = outputs1[1]
        T2 = outputs2[1]
    elif layer=='conv2':
        T1 = outputs1[2]
        T2 = outputs2[2]
    elif layer=='conv3':
        T1 = outputs1[3]
        T2 = outputs2[3]
    elif layer=='conv4':
        T1 = outputs1[4]
        T2 = outputs2[4]
    else:
        raise Exception('Layer parameter has unrecognized value')

    # flatten the tensors for each image
    T1 = T1.data.cpu().numpy().reshape(N,-1)
    T2 = T2.data.cpu().numpy().reshape(N,-1)

    v_sim = np.zeros(N)
    for i in range(N):
        v1 = T1[i,:]
        v2 = T2[i,:]
        v_sim[i] = 1-cosine(v1,v2) # using cosine distance
    return v_sim
