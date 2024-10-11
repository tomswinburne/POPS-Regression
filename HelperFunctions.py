import numpy as np
from scipy.stats import truncnorm
import matplotlib.pyplot as plt

def TestModels(N_train=2000, N_test=1000, P=15, seed=123,\
                model='sinusoidal',noise=0.0):
    """
    Generate test data for deterministic regression problems.

    In all cases ${\bf X}$ is truncnormal(-3,3)
    
    `specified`:
    $$
    E({\bf X}) = {\bf m}_l\cdot{\bf X} / P 
    $$

    `sinusoidal`:
    $$
    E({\bf X}) = {\bf m}_l\cdot{\bf X} / P + \sin(2\pi{\bf m}_1\cdot{\bf X})
    $$

    `random`:
    Randomly pick a linear model:
    $$
    E({\bf X}) = {\bf m}_l\cdot{\bf X} / P,\quad l\sim[0,M)
    $$

    `quadratic`:
    Add a quadratic perturbation:
    $$
    E({\bf X}) = {\bf m}_0\cdot{\bf X} / P + ({\bf m}_1\cdot{\bf X}/P)^2
    $$

    `cubic`:
    Add a cubic perturbation:
    $$
    E({\bf X}) = {\bf m}_0\cdot{\bf X} / P + ({\bf m}_1\cdot{\bf X}/P)^3
    $$

    Parameters
    ----------
    N_train : int, optional
        Number of training data, by default 2000
    N_test : int, optional
        Number of test data, by default 1000
    P : int, optional
        Parameters, by default 15
    seed : int, optional
        Random number seed, by default 123
    model : str, optional
        Model type, by default 'sinusoidal', one of ['noise', 'sinusoidal', 'random', 'cubic', 'quadratic']
    noise : float, optional
        Add additional noise term to scale ${\bf\eta},\quad{\bf\eta}\sim\mathcal{N}({\bf 0},\mathbb{I})$, by default 0.0
    
    Returns
    -------
    dict
        A dictionary containing the following results:
        - 'Theta_LST_SQ': Least squares solution
        - 'MAE_LST_SQ': Mean absolute error of the least squares solution
        - 'MAE_POINTWISE_Gauss': Mean absolute error of the pointwise Gaussian fit
        - 'MAE_MEAN_POINTWISE_BOUND': Mean absolute error of the mean pointwise bound
        - 'MAE_MAX_POINTWISE_BOUND': Mean absolute error of the maximum pointwise bound
        - 'MAE_POINTWISE_FALSE_POSITIVE': Percentage of false positives in the pointwise bound
        - 'HIST_AE_LST_SQ': Histogram of absolute errors for the least squares solution
        - 'HIST_AE_BOUND': Histogram of absolute errors for the pointwise bound
        - 'LEVERAGE_TEST': Leverage of the test data
        - 'LEVERAGE_TRAIN': Leverage of the training data
        - 'X_TRAIN': Training data input
        - 'X_TEST': Test data input
        - 'Y_TRAIN': Training data output
        - 'Y_TEST': Test data output
        - 'DESIGN_MAT': Design matrix
        - 'ALL_POINTWISE_FITS': All pointwise fits

    Raises
    ------
    AssertionError
        If the model type is not one of ['noise', 'sinusoidal', 'random', 'cubic', 'quadratic']

    """
    assert model in ['noise', 'sinusoidal', 'random', 'cubic', 'quadratic']
    
    P = P - 1
    N_total = N_train + N_test

    # random correlated input signals
    np.random.seed(seed)
    # +/- 3 sigma
    G = lambda X, Y: truncnorm.rvs(-3.0, 3.0, size=(X, Y))

    X = np.hstack((G(N_total, P), np.ones((N_total, 1))))
    X_test = X[N_train:]
    X_train = X[:N_train]
    P += 1

    if model == 'cubic':
        model_name = "Cubic engine"
        # random model
        m_l = np.random.uniform(-1., 1., size=P) / np.sqrt(P)  # linear
        m_l -= m_l.mean()

        m_q = np.random.uniform(-1., 1., size=P) / np.sqrt(P)  # non-linear
        m_q -= m_q.mean()

        E = lambda X: X @ m_l + (X @ m_q)**3

    elif model == 'sinusoidal':
        model_name = "Sinusoidal engine"
        # random model
        m_l = np.random.uniform(-1., 1., size=P) / (P)  # linear
        m_l -= m_l.mean()

        m_s = np.random.uniform(-1., 1., size=P)  # non-linear
        
        E = lambda X: X @ m_l + np.sin(X @ m_s * np.pi)

    elif model == 'quadratic':
        model_name = "Quadratic engine"

        # random model
        m_l = np.random.uniform(-1., 1., size=P) / (P)  # linear
        m_l -= m_l.mean()

        m_s = np.random.uniform(-1., 1., size=P) / (P)  # non-linear
        m_s -= m_s.mean()

        E = lambda X: X @ m_l + (X @ m_s)**2 

    else:
        model_name = "Random linear engine"
        # random model
        m = np.random.uniform(-1., 1., size=(max(2, N_total // 40), P))
        m -= np.outer(m.mean(1), np.ones(m.shape[1]))
        #m *= 0.2
        m += np.random.uniform(-1., 1., size=P)
        m /= P
        E = lambda X: (X * m[np.random.randint(m.shape[0], size=X.shape[0])]).sum(1)
    
    if noise>0.0:
        E_final = lambda X: E(X) + np.random.uniform(-0.5,0.5,size=X.shape[0]) * noise
    else:
        E_final = E 
    
    # outputs
    y_train = E_final(X_train)
    y_test = E_final(X_test)

    # leverage
    w = np.ones(N_train) / N_train
    A = np.linalg.inv(np.einsum('ij,i,ik->jk', X_train, w, X_train))
    leverage = np.einsum('ij,ij->i', X_train @ A, X_train)

    """
    # leverage-weighted fits
    w = 1.0 / leverage
    w /= w.sum()
    A = np.linalg.inv(np.einsum('ij,i,ik->jk', X_train, w, X_train))
    leverage = np.einsum('ij,ij->i', X_train @ A, X_train)
    """

    m_lst_sq = A @ np.einsum('i,ij->j', y_train * w, X_train)
    abs_err = np.abs(y_test - X_test @ m_lst_sq)
    
    # pointwise fits
    err_train = y_train - X_train @ m_lst_sq
    pointwise = np.einsum('i,ij->ij', err_train / leverage, X_train @ A)
    
    mae_gauss_pointwise = (X_train @ np.cov(pointwise.T) * X_train).sum(1).mean() 
    mae_gauss_pointwise = np.sqrt(2.0 * P * mae_gauss_pointwise / np.pi)
    
    bound = np.abs(X_test @ pointwise.T).max(1)
    mae_err_pointwise = np.abs(X_test @ pointwise.T).mean()

    # self-consistency
    assert np.abs(err_train - np.diag(X_train @ pointwise.T)).max() < 1.0e-9

    res = {}
    res['Theta_LST_SQ'] = m_lst_sq.copy()
    res['MAE_LST_SQ'] = abs_err.mean()
    res['MAE_POINTWISE_Gauss'] = mae_gauss_pointwise
    res['MAE_MEAN_POINTWISE_BOUND'] = mae_err_pointwise.mean()
    res['MAE_MAX_POINTWISE_BOUND'] = bound.mean()
    res['MAE_POINTWISE_FALSE_POSITIVE'] = 100. * (bound < abs_err).mean()
    res['MODEL'] = model_name
    res['HIST_AE_LST_SQ'] = np.histogram(abs_err, bins=101, density=True)
    res['HIST_AE_BOUND'] = np.histogram(np.abs(X_test @ pointwise.T), bins=101, density=True)

    res['LEVERAGE_TEST'] = np.einsum('ij,ij->i', X_test @ A, X_test)
    res['LEVERAGE_TRAIN'] = np.einsum('ij,ij->i', X_train @ A, X_train)
    res['X_TRAIN'] = X_train
    res['X_TEST'] = X_test
    res['Y_TRAIN'] = y_train
    res['Y_TEST'] = y_test
    res['DESIGN_MAT'] = A
    res['ALL_POINTWISE_FITS'] = pointwise + m_lst_sq

    all_errs = X_train@(pointwise + m_lst_sq).T - y_train[:,None]
    var_err = (y_train - X_train @ m_lst_sq).var()
    m_ij = np.exp(-all_errs**2/var_err * 10.0).T
    
    res['ALL_POINTWISE_WEIGHTS'] = (m_ij / m_ij.mean(0)[:,None]).mean(1)
    res['ALL_POINTWISE_WEIGHTS'] /= res['ALL_POINTWISE_WEIGHTS'].mean()

    evals,evec = \
        np.linalg.eigh(res['ALL_POINTWISE_FITS'].T@res['ALL_POINTWISE_FITS'])
    evals /= evals.max()
    R = (evals>1.0e-6).sum()
    projector = evec[:,evals>1.0e-6] # P->R projector
    res['HYPERCUBE_PROJECTOR'] = projector.copy()
    res['HYPERCUBE_LOWER'] = (res['ALL_POINTWISE_FITS']@projector).min(0)
    res['HYPERCUBE_MAX'] = (res['ALL_POINTWISE_FITS']@projector).max(0)
    res['HYPERCUBE_RANGE'] = res['HYPERCUBE_MAX'] - res['HYPERCUBE_LOWER']
    res['HYPERCUBE_CENTER'] = res['HYPERCUBE_LOWER']+0.5*res['HYPERCUBE_RANGE']
    res['HYPERCUBE_MEAN'] = res['HYPERCUBE_CENTER']@projector.T # (P,)
    res['HYPERCUBE_VARIANCE'] = \
        projector@np.diag(res['HYPERCUBE_RANGE']**2)@projector.T / 12.0 # (P,P)
    
    res['HYPERCUBE_SAMPLES'] = \
        np.random.uniform(low=res['HYPERCUBE_LOWER'],\
            high=res['HYPERCUBE_LOWER']+res['HYPERCUBE_RANGE'],
            size=(y_train.size,R))@projector.T
    
    def hypercube_bound(X):
        vx = X@res['HYPERCUBE_PROJECTOR'] # N,R
        vx_low = vx@res['HYPERCUBE_LOWER']
        vx *= res['HYPERCUBE_RANGE'][None,:]
        vx_max = vx_low+np.where(vx>0.,vx,0.).sum(1)
        vx_min = vx_low+np.where(vx<0.,vx,0.).sum(1)
        return vx_min,vx_max 
    
    def hypercube_face_bound(X):
        vx = X@res['HYPERCUBE_PROJECTOR'] # N,R
        vx_low = vx@res['HYPERCUBE_LOWER']
        vx *= res['HYPERCUBE_RANGE'][None,:]
        #vx_low += vx.sum(1)
        vx_max = vx_low+vx.max(1)
        vx_min = vx_low+vx.min(1)
        return vx_min,vx_max 
    
    res['HYPERCUBE_BOUND_FN'] = hypercube_bound
    res['HYPERCUBE_FACE_BOUND_FN'] = hypercube_face_bound
        
    assert np.abs(err_train - np.diag(X_train @ pointwise.T)).max() < 1.0e-9

    return res

def uniform_resample(M,N_resample=10):
    """
        Generate a uniform resampling, replacing second axis 
    """
    resampled = np.random.uniform(-.5,.5,size=(M.shape[0],N_resample))
    resampled *= M.ptp(1)[:,None]
    resampled += ((M.min(1)+M.max(1))/2.0)[:,None]
    return resampled

def hist(ax,data,w=None,label=None,env_pc=None,bins=50,color='k'):
    """
        Plot a histogram of data on axis ax, with label label
    """
    mae = int(np.round(np.abs(data).mean(),2)*100.-100.0)
    
    if label is not None:
        label += f' MAE: {mae}%'
        if env_pc is not None:
            label += f', EV: {env_pc}%'
    
    return ax.hist(data,bins=bins,weights=w,\
            density=True,cumulative=False,\
            histtype='step',lw=2,\
            label=label,
            color=color
            )[0].max();


def envelope(ens_errors,errors,label='raw'):
    """
        Plot envelope statistics of ensemble errors
    """
    mae = np.abs(errors).mean()
    print(f'MAE {label}',(np.abs(errors)/np.abs(ens_errors).mean(1)).mean())
    print(f'RMS {label}',(np.abs(errors)**2/(ens_errors**2).mean(1)).mean())
    print(f'ENV {label}',(np.abs(errors)/ens_errors.ptp(1)).mean())

    envelope = (ens_errors.max(1)<errors)+(ens_errors.min(1)>errors)
    env_pc = np.round(envelope.mean()*100,1)
    print(f"Envelope Violation ({label}):",env_pc,"%")
    return env_pc

def envelope_plot(mle,errors,env_pc,label,ax,color='k'):
    """
        Plot envelope statistics of ensemble errors
    """
    ax.fill_between(mle,\
        errors.min(1)+mle,\
        errors.max(1)+mle,\
        alpha=0.5,\
        label=f'{label} ({100-env_pc}%)',
        facecolor=color,)

def PlotModel(res,axs=None,filename=None):
    if axs is None:
        axs = plt.subplots(1,2,figsize=(6,2),dpi=160)

    N = res['X_TRAIN'].shape[0]
    P = res['X_TRAIN'].shape[1]
    Thetas = res['ALL_POINTWISE_FITS']
    weights = res['ALL_POINTWISE_WEIGHTS']
    # testing....
    #thresh = np.percentile(weights,90)
    #weights[weights<thresh] = 0.0
    weights /= weights.sum()

    Thetas_HC = res['HYPERCUBE_SAMPLES']

    # test MLE predictions and order by prediction
    mle = res['X_TEST']@Thetas.mean(0)
    errors = (res['Y_TEST'] - mle)
    o = mle.argsort()
    
    # reorder
    errors = errors[o]
    X_test = res['X_TEST'][o]
    mle = mle[o]
    y_test = res['Y_TEST'][o]

    # mle errors
    true_mae = np.abs(errors).mean()
    errors /= true_mae
    y_test /= true_mae
    mle /= true_mae
    mae = 1.0
    # histograms
    bins = np.linspace(-1.5,1.5,100) * np.abs(errors).max()

    lim = hist(axs[0],errors,label=None,bins=bins,color='k')
    axs[1].plot(mle,y_test,'.',color='k',label=None)


    # ensemble ansatz
    all_errors = X_test @ (Thetas.T - Thetas.mean(0)[:,None]) / true_mae
    
    env_pc = envelope(all_errors,errors,'raw')
    N_resample = Thetas.shape[0] * 10
    c='C1'
    ens_select = np.random.choice(Thetas.shape[0],N_resample,\
        replace=True,p=weights)
    
    all_errors = all_errors[:,ens_select]
    all_errors -= all_errors.mean()
    
    env_pc = envelope(all_errors,errors,'Resampled')
    
    envelope_plot(mle,all_errors,env_pc,r'$\pi^*_E$ resample',axs[1],c)

    lim = max(lim,hist(axs[0],all_errors.flatten(),\
                    label=r'$\pi^*_E$,',env_pc=env_pc,bins=bins,color=c))

    # parameter resampling
    c='C2'
    random_errors = X_test @ (Thetas_HC.T - Thetas.mean(0)[:,None]) / true_mae
    random_errors -= random_errors.mean()

    env_pc = envelope(random_errors,errors,'Random')
    envelope_plot(mle,random_errors,env_pc,\
        r'$\pi^*_\mathcal{H}$ resample',axs[1],c)
    lim = max(lim,hist(axs[0],random_errors.flatten(),\
                    label=r'$\pi^*_\mathcal{H}$,',env_pc=env_pc,bins=bins,color=c))


    axs[0].set_ylim(0.0,lim*1.5)
    #axs[1].set_ylim(all_errors.min(),all_errors.max())
    
    axs[1].set_title("Error [Loss MAE]",fontsize=9)
    axs[1].set_xlabel("Min. Loss Prediction",fontsize=9)
    #axs[0].set_title("P(x< Error < x+dx)",fontsize=9)
    axs[0].set_xlabel("Error / Loss MAE",fontsize=9)
    axs[1].set_ylabel("Prediction / Loss MAE",fontsize=9)
    axs[0].set_title(rf'P(Error$\in$[x,x+dx]), P={P}, N/P={N//P}',fontsize=9)
    axs[1].set_title('Prediction Envelopes',fontsize=9)
    axs[0].legend(loc='upper left',fontsize=8)
    axs[1].legend(loc='upper left',fontsize=8)

    #plt.suptitle(f"{res['MODEL']}, P={P}, N/P={N//P}")
    plt.tight_layout()
    if filename is not None:
        plt.savefig(filename)

