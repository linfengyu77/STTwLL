import numpy as np

def omp_n(D, Y, K):
    '''
    D: dictionary
    Y: observations
    K: sparsity
    '''
    nA = D.shape[1]
    nY = Y.shape[1]
    X  = np.zeros((nA, nY))

    # D = D @ np.diag(np.sqrt(1/np.diag(D.T @ D)))

    for n in range(nY):
        y = Y[:, n, None]
        x = np.zeros((nA, 1))
        ai = []
        for _ in range(K):
            r = y - D @ x
            aProj = np.abs(D.T @ r)
            aProj[ai] = -1
            I = np.argmax(aProj)
            ai.extend([I])
            xp = np.linalg.pinv(D[:, ai]) @ y
            x[ai] = xp
        X[:, n] = x.flatten()
    # R = Y - D @ X
    return X



