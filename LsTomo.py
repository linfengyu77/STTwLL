import numpy as np
import matplotlib.pyplot as plt
np.random.seed(123)

def conventional_tomo1(eta, L, tomoMtrix, refSlowess, 
            travelTime, sTrue, validBounds=None, normNoise=None, noiseRelize=None, plot=False):
    refSlowess = np.squeeze(refSlowess)
    w1, w2 = sTrue.shape[0], sTrue.shape[1]
    xxc, yyc = np.meshgrid(range(w1), range(w2))
    npix = w1 * w2
    sig_L = np.zeros((npix, npix)) 
    xxc1 = xxc.flatten()
    yyc1 = yyc.flatten()
    for ii in range(npix):
        distc = np.sqrt((xxc - xxc1[ii]) ** 2 + (yyc - yyc1[ii]) **2 )
        distc = distc.flatten()
        sig_L[ii, :] = np.exp(-distc/L)
    
    invsig_L = np.linalg.solve(sig_L, np.eye(npix)) # np.linalg.pinv(sig_L) @ np.eye(npix)

    # invert for slowness
    Tref = tomoMtrix @ (refSlowess * np.ones((npix,1)))

    # time perturbation
    dT = travelTime - Tref

    G = tomoMtrix.T @ tomoMtrix + eta * invsig_L
    ds = np.linalg.solve(G, tomoMtrix.T) @ dT # np.linalg.pinv(G) @ G.T @ dT

    if plot == True:
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)
        im = ax1.imshow((np.reshape(ds+refSlowess, sTrue.shape) * validBounds))
        ax1.set_xlabel("Range (km)")
        ax1.set_ylabel("Range (km)")
        plt.colorbar(im)
    return ds



def conventional_tomo2(eta, L, tomoMtrix, refSlowess, 
            travelTime, sTrue, validBounds=None, normNoise=None, noiseRelize=None, plot=False):
    # refSlowess = np.squeeze(refSlowess)
    w1, w2 = sTrue.shape
    xxc, yyc = np.meshgrid(range(1, w1+1), range(1, w2+1))
    npix = w1 * w2
    sig_L = np.zeros((npix, npix)) 
    xxc1 = xxc.flatten()
    yyc1 = yyc.flatten()
    for ii in range(npix):
        distc = np.sqrt((xxc - xxc1[ii]) ** 2 + (yyc - yyc1[ii]) **2 )
        distc = distc.flatten()
        sig_L[ii, :] = np.exp(-distc/L)
    
    invsig_L = np.linalg.solve(sig_L, np.eye(npix)) # np.linalg.pinv(sig_L) @ np.eye(npix)

    # invert for slowness
    Tref = tomoMtrix @ (refSlowess * np.ones((npix, 1)))

    # time perturbation
    dT = travelTime - Tref
    G = tomoMtrix.T @ tomoMtrix + eta * invsig_L
    ds = np.linalg.solve(G, tomoMtrix.T) @ dT # np.linalg.pinv(G) @ G.T @ dT

    if plot == True:
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(111)
        im = ax1.imshow((np.reshape(ds+refSlowess, sTrue.shape) * validBounds))
        ax1.set_xlabel("Range (km)")
        ax1.set_ylabel("Range (km)")
        plt.colorbar(im)
        plt.show()

    return ds



