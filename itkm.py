import numpy as np



def itkm(data, k, s, maxit=30):
    '''
    data: d x n matrix containing signals
    k: number of atoms
    s: sparsity level
    maxit: mumber of iteratons - default 1000
    dinit: initialisaztion, d x k norm colum matrix -default
    '''
    d, N = data.shape
    dinit = np.random.randn(d, k)
    scale = np.sum(dinit * dinit, axis=0)
    dinit = dinit @ np.diag(1/np.sqrt(scale))
    dold = np.copy(dinit)

    for _ in range(maxit):
        ip = dold.T @ data
        absip  = np.abs(ip)
        signip = np.sign(ip)
        I = np.argsort(-absip, axis=0, kind='stable')
        dnew = np.zeros((d, k))
        for n in range(N):
            dnew[:, I[0:s, n]] = dnew[:, I[0:s, n]] + data[:, n, None] @ signip[I[0:s, n], n, None].T
        scale = np.sum(dnew * dnew, axis=0)
        nonzero = scale > 0.001
        dnew[:, nonzero] = dnew[:, nonzero] @ np.diag(1/np.sqrt(scale[nonzero]))
        dold[:, nonzero] = dnew[:, nonzero]
        
    return dold
        

