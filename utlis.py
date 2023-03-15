from secrets import choice
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import math
import random
from turtle import back
import torch
import itertools
import scipy.sparse as ss
from matplotlib.path import Path
from matplotlib.colors import Normalize, Colormap
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
# from skimage.measure import compare_ssim, compare_psnr, compare_mse
from scipy import ndimage
from mpl_toolkits.axes_grid1 import make_axes_locatable
# random.seed(1234)
# np.random.seed(1234)


def extract_patches2d(x, kernel_size=17, stride=7, padding=0):
    kernel_size = _pair(kernel_size)
    stride = _pair(stride)
    padding = _pair(padding)
    if padding != (0,0):
        x = F.pad(x, [padding[1], padding[1], padding[0], padding[0]])
    x = x.unfold(2, kernel_size[0], stride[0]).unfold(3, kernel_size[1], stride[1])
    x = x.permute(2,3,0,1,4,5).reshape(-1, *kernel_size)
    return x

def sample_patches(images, ntrain, ntest, **kwargs):
    # extract image patches
    patches = extract_patches2d(images, **kwargs)
    
    # remove empty patches
    patches = patches[patches.sum([1,2]) > 0]
    
    # select train/test examples
    assert images.size(0) >= ntrain + ntest, 'not enough data for the desired ntrain/ntest'
    idx = torch.randperm(patches.size(0))
    
    return patches[idx[:ntrain]], patches[idx[ntrain:ntrain+ntest]]

def plot_filter(ax, W, Wmag, cmap):
    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)
    assert isinstance(cmap, Colormap)
    W = Normalize()(W)
    img = cmap(W)
    if Wmag is not None:
        img[..., -1] = Wmag
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])

def compute_magnitudes(W, pct=1.):
    mag = W.flatten(1).norm(dim=1)
    mag /= torch.quantile(mag, pct)
    return mag.clamp(0, 1)

def show_filters(W, ncol=24, magnitudes=True, mag_pct=1., size=0.7,
                 cmap='viridis', title=None, hspace=0, wspace=0):
    if W.dim() == 4 and W.size(1) == 1:
        W = W.squeeze(1)  # handle pytorch format
    n = W.size(0)
    nrow = math.ceil(n / ncol)
    Wmag = compute_magnitudes(W, mag_pct) if magnitudes else n * [None]

    fig, axes = plt.subplots(nrow, ncol, figsize=(size * ncol, size * nrow))
    axes = axes.ravel()
    for i in range(nrow * ncol):
        if i >= n:
            axes[i].set_axis_off()
            continue
        plot_filter(axes[i], W[i], Wmag[i], cmap)
    if title is not None:
        plt.suptitle(title)
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.show()



def view_results(weight, n=None, show_norms=False, **kwargs):
    W = weight.detach().cpu().T
    norms, ix = W.norm(dim=1).sort(descending=True)
    W = W[ix].reshape(W.size(0), 17, 17)
    if n is not None:
        norms, W = norms[:n], W[:n]
        
    # show bar plot of filter norms
    if show_norms:
        norms = norms[:50]
        plt.figure(figsize=(10,3))
        plt.bar(range(len(norms)), norms)
        plt.show()
        
    # visualize filters
    show_filters(W, **kwargs)

def random_choice(x, n, dim=0):
    N = x.size(dim)
    ix = torch.randperm(N)[:n]
    return x.index_select(dim, ix)

def normalization(x):
    mu = np.mean(x)
    std = np.std(x)
    return (x - mu)/std

def rmse(y, pred, mask):
    return np.sqrt(((y*mask - pred*mask) ** 2).sum()/np.sum(mask))*1000


def sub2ind(array_shape, rows, cols):
    ind = rows * array_shape[1] + cols
    ind[ind < 0] = 0
    ind[ind >= array_shape[0]*array_shape[1]] = array_shape[0]*array_shape[1]
    return ind.astype(int)


def survey_deisgn(nStaion, sTrue, save_path=None):
    stas = np.random.randn(nStaion, 2)
    stas = np.sign(stas) * abs(stas) ** (7/8)
    staRation = 49 / np.max(abs(stas), axis=0)
    xSta = stas[:, 0] * staRation[0] + 49
    ySta = stas[:, 1] * staRation[1] + 49

    staPerms = np.transpose(list(itertools.combinations(list(range(nStaion)), 2)))
    len_staPerms = np.max(staPerms.shape)

    staPermsVec = staPerms.reshape(-1, 1)

    xStaChoose = xSta[staPermsVec]
    yStaChoose = ySta[staPermsVec]

    xStaResh = np.reshape(xStaChoose, [2, len_staPerms])
    yStaResh = np.reshape(yStaChoose, [2, len_staPerms])

    X1 = np.c_[xStaResh[0, :], yStaResh[0, :]]
    X2 = np.c_[xStaResh[1, :], yStaResh[1, :]]

    bps= np.array([[7.7189,   39.5190],
                [16.2442,   21.4140],
                [31.6820,   14.0671],
                [45.7373,   11.7055],
                [70.8525,   16.6910],
                [91.3594,   22.9883],
                [97.8111,   44.2420],
                [88.1336,   66.5452],
                [70.8525,   77.5656],
                [36.5207,   97.7697],
                [28.6866,   89.6356]])

    x, y = np.meshgrid(range(1, 101), range(1, 101))
    x, y = x.reshape(-1, 1), y.reshape(-1, 1)
    b_p = np.hstack((x, y))
    path = Path(bps)
    vb  = path.contains_points(b_p)
    vb  = 1 * vb
    vb2 = ndimage.binary_dilation(vb, iterations=5).astype(vb.dtype)


    # calculating rays and travel travel time and sensing matrix
    mult = 100                          # ensuring enough resolution for ray steps (can increase this number to improve accuracy)
    sc = sTrue.shape
    lc = sTrue.size
    steps = math.ceil(1.5 * np.max(sc))            # step size for ray propagation
    av = []
    ii = []
    jj = []

    for m in range(X1.shape[0]):
        src = X1[m, :, None]
        rec = X2[m, :, None]
        # finding ray trajectory
        v0 = rec - src
        vn = np.linalg.norm(v0)
        v  = v0 / vn                    #ray vector (normalized)
     
        ls = np.linspace(0, steps, steps*mult, endpoint=True)
        # tracing ray
        points = src + v @ ls.reshape(1, -1)      #stepping through ray trajectory, number of points should be large, includes endpoints
        pcheck = np.sum((points - rec) ** 2, axis=0, keepdims=True)                               # finding intersection of ray with pixel
        # u = np.min(pcheck)
        i = np.argmin(pcheck)
        inds   = np.floor(points[:, 1:i])      # starts @ 2 since ray travels from source to receiver
        npoints = np.max(inds.shape)

        xs = inds[0, :] + 1
        ys = inds[1, :] + 1
        ds_int = (vn / npoints)                # length of ray/#points
            
        indA = sub2ind(sc, xs, ys)
        # idx = np.c_[xs, ys]
        # idx = idx.astype(int)
        # indA = np.ravel_multi_index(idx.T, sc)
        
        a0=np.zeros(lc)
        for k in range(xs.size):
            a0[indA[k]] = a0[indA[k]] + ds_int # one tomo matrix row

        colInds = np.unique(indA)
        av.extend(a0[colInds])           # tomo mtx vector, much faster than indexing array
        ii.extend(m * np.ones(colInds.size))
        jj.extend(colInds)

    av = np.array(av)
    ii = np.array(ii)
    jj = np.array(jj)

    A = ss.csr_matrix((av.flatten(), (ii.flatten(), jj.flatten())), shape=(np.max(X1.shape), sTrue.size))
    A = A.todense()
    Tarr = A @ sTrue.reshape(-1, 1)

    # Plot slowness map and rays
    if save_path is not None:
        extent = 0, 100, 0, 100
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        fig = plt.figure(figsize=(9, 4))
        ax1 = fig.add_subplot(121)
        im1 = ax1.imshow(sTrue, extent=extent)
        ax1.set_xlabel("Range (km)", fontsize=14)
        ax1.set_ylabel("Range (km)", fontsize=14)
        # plt.vlines(35, 0, 100, colors='k')                                  # for Marmousi model only
        plt.tick_params(top=True,bottom=True,left=True,right=True)
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax, fraction=0.046)        

        ax2 = fig.add_subplot(122)
        for m in range(len(X1)):
            ax2.plot([X1[m, 0], X2[m, 0]], [X1[m, 1], X2[m, 1]],\
                 color='k', marker='+', markeredgecolor = 'r',linewidth=1)
        plt.rcParams['xtick.direction'] = 'in'
        plt.rcParams['ytick.direction'] = 'in'
        plt.tick_params(top=True,bottom=True,left=True,right=True)                                       
        ax2.margins(0)
        
        # raysPerPix = np.reshape(np.sum(A, axis=0), sTrue.shape)          # calculating ray density

        # ax3 = fig.add_subplot(133)
        # im2 = ax3.imshow(np.log10(raysPerPix, out=np.zeros_like(raysPerPix), where=(raysPerPix!=0)))
        # ax3.set_xlabel("Range (km)")
        # ax3.set_ylabel("Range (km)")
        # ax3.margins(0)
        # plt.colorbar(im2, fraction=0.046)        
        fig.tight_layout()
        plt.show()
        fig.savefig(save_path + 'slowness_rays.pdf')
    
    return np.array(Tarr), np.array(A), np.array(vb.reshape(-1, 1)), vb2.reshape(-1, 1)

def checkerboard(boardsize, squaresize):
    return np.fromfunction(lambda i, j: (i//squaresize[0])%2 != (j//squaresize[1])%2, boardsize).astype(int)


def slownessMap(type):
    # function for generating synthetic slowness maps
    # INPUT: 'type'= 'sd' for smooth-discontinuous
    
    if type == 'sd':
        # smooth variation with discontinuity
        ds_maskBar = np.zeros((100,100))
        ds_maskBar[:, 47:53] = 1
        # ds_maskBar[60:66, :] = 1
        xx, yy = np.meshgrid(range(1, 101), range(1, 101))
        ds = (np.sin(2*np.pi*(xx-10)/50) + np.sin(2*np.pi*(yy+10)/50))/2   #   -sqrt(2)/2; % sinusoidal slowness
        ds = ds * 0.1    # scaling sinusoid
        ds_bar = ds * ds_maskBar
        ds = ds - ds_bar + 0.1*ds_maskBar
    else:
        print('invalid type!')
    s = ds + 0.4                 # adding perturbations to background speeds
    return s

def patchSamp(A, blocks):
    nPatch, nBlocks = blocks.shape
    Bpass = np.zeros_like(blocks)
    for k in range(nBlocks):
        Bpass[:, k] = np.sum(A[:, blocks[:, k]], axis=0)
    Bpass_zero = Bpass != 0
    zeroCheck = np.sum(Bpass_zero, axis=0)
    percZero  = zeroCheck/nPatch
    return percZero

def getPatches(w1, w2, nib):
    im = np.arange(w1*w2).reshape(w1, w2, order='F')
    imTop  = im[0:nib-1, :]
    imLeft = im[:, 0:nib-1]
    imUL   = imTop[0:nib-1, 0:nib-1]
    imInds = np.vstack((np.hstack((im, imLeft)), np.hstack((imTop, imUL))))
    patches = im2col(imInds, [nib,nib])
    return patches


def im2col(mtx, block_size):
    mtx_shape = mtx.shape
    sx = mtx_shape[0] - block_size[0] + 1
    sy = mtx_shape[1] - block_size[1] + 1
    # 如果设A为m×n的，对于[p q]的块划分，最后矩阵的行数为p×q，列数为(m−p+1)×(n−q+1)。
    result = np.empty((block_size[0] * block_size[1], sx * sy), dtype=int)
    # 沿着行移动，所以先保持列（i）不动，沿着行（j）走
    for i in range(sy):
        for j in range(sx):
            result[:, i * sx + j] = mtx[j:j + block_size[0], i:i + block_size[1]].ravel(order='F')
    return result


def col2im(mtx, image_size, block_size):
    p, q = block_size
    sx = image_size[0] - p + 1
    sy = image_size[1] - q + 1
    result = np.zeros(image_size)
    weight = np.zeros(image_size)  
    col = 0

    for i in range(sy):
        for j in range(sx):
            result[j:j + p, i:i + q] += mtx[:, col].reshape(block_size, order='F')
            weight[j:j + p, i:i + q] += np.ones(block_size)
            col += 1
    return result / weight

def dct_dict(n, m):
    """ dct dictionary 
        n: patch_size
        m: sqrt(atoms)
    """
    Dictionary = np.zeros((n, m))
    for k in range(m):
        V = np.cos(np.arange(0, n) * k * np.pi / m)
        if k > 0:
            V = V - np.mean(V)
        Dictionary[:, k] = V / np.linalg.norm(V)
    Dictionary = np.kron(Dictionary, Dictionary)
    Dictionary = Dictionary.dot(np.diag(1 / np.sqrt(np.sum(Dictionary ** 2, axis=0))))
    idx = np.arange(0, n ** 2)
    idx = idx.reshape(n, n, order="F")
    idx = idx.reshape(n ** 2, order="C")
    Dictionary = Dictionary[idx, :]
    return Dictionary


def normalize(image):
    """
    Maps first to [0, image.max() - image.min()]
    then to [0, 1]

    :param image:
        numpy.ndarray

    :return:
        image mapped to [0, 1]
    """
    image = image.astype(float)
    if image.min() != image.max():
        image -= image.min()
    
    nonzeros = np.nonzero(image)
    image[nonzeros] = image[nonzeros] / image[nonzeros].max()
    return image

def vd(dictionary, rows, cols, prefix, show=True, save=False, title=None):
    """
        Visualize a dictionary

        :param dictionary:
            Dictionary to plot

        :param rows:
            Number of rows in plot

        :param cols:
            Number of columns in plot. Rows*cols has to be equal
            to the number of atoms in dictionary

        :param show:
            Call pyplot.show() is True

        :param title:
            Title for figure
    """
    size = int(np.sqrt(dictionary.shape[0])) + 2
    img = np.zeros((rows * size, cols * size))

    for row in range(rows):
        for col in range(cols):
            atom = row * cols + col
            at = normalize(dictionary[:, atom].reshape(size - 2, size - 2))
            padded = np.pad(at, 1, mode='constant', constant_values=1)
            img[row * size:row * size + size, col * size:col * size + size] = padded
    fig = plt.figure()
    plt.imshow(img, cmap=plt.cm.bone, interpolation='nearest')
    plt.tight_layout()
    plt.axis('off')

    if title is not None:
        title = title if isinstance(title, str) else str(title)
        plt.title(title)

    if show:
        plt.show()
    
    if save:
        fig.savefig(save + 'The_learned_dictionary_by_{}.pdf'.format(prefix))
        




