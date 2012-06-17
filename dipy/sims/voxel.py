import numpy as np
from dipy.core.geometry import sphere2cart, cart2sphere
from IPython import embed

def SticksAndBall(bvals,gradients,d=0.0015,S0=100,angles=[(0,0),(90,0)],fractions=[35,35],snr=20):
    """ Simulating the signal for a Sticks & Ball model 
    
    Based on the paper by Tim Behrens, H.J. Berg, S. Jbabdi, "Probabilistic Diffusion Tractography with multiple fiber orientations
    what can we gain?", Neuroimage, 2007. 
    
    Parameters
    -----------
    bvals : array, shape (N,)
    gradients : array, shape (N,3) also known as bvecs
    d : diffusivity value 
    S0 : unweighted signal value
    angles : array (K,2) list of polar angles (in degrees) for the sticks
        or array (K,3) with sticks as Cartesian unit vectors and K the number of sticks
    fractions : percentage of each stick
    snr : signal to noise ration assuming gaussian noise. Provide None for no noise.
    
    Returns
    --------
    S : simulated signal
    sticks : sticks in cartesian coordinates 
    
    """
    
    fractions=[f/100. for f in fractions]    
    f0=1-np.sum(fractions)    
    S=np.zeros(len(gradients))
    
    angles=np.array(angles)
    if angles.shape[-1]==3:
        sticks=angles
    if angles.shape[-1]==2:
        sticks=[ sphere2cart(1,np.deg2rad(pair[0]),np.deg2rad(pair[1]))  for pair in angles]    
        sticks=np.array(sticks)
    
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=f0*np.exp(-bvals[i+1]*d)+ np.sum([fractions[j]*np.exp(-bvals[i+1]*d*np.dot(s,g)**2) for (j,s) in enumerate(sticks)])
        S[i+1]=S0*S[i+1]    
    S[0]=S0    
    if snr!=None:
        std=S0/snr
        S=S+np.random.randn(len(S))*std
    
    return S,sticks

def SingleTensor(bvals,gradients,S0,evals,evecs,snr=None):
    """ Simulated signal with a Single Tensor
     
    Parameters
    ----------- 
    bvals : array, shape (N,)
    gradients : array, shape (N,3) also known as bvecs
    S0 : double,
    evals : array, shape (3,) eigen values
    evecs : array, shape (3,3) eigen vectors
    snr : signal to noise ratio assuming gaussian noise. 
        Provide None for no noise.
    
    Returns
    --------
    S : simulated signal
    
    
    """
    S=np.zeros(len(gradients))
    D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T)    
    #print D.shape
    for (i,g) in enumerate(gradients[1:]):
        S[i+1]=S0*np.exp(-bvals[i+1]*np.dot(np.dot(g.T,D),g))
    S[0]=S0
    if snr!=None:
        std=S0/snr
        S=S+np.random.randn(len(S))*std
    return S
    
def Kurtosis4D(kunique):
    """Returns full kurtosis tensor given its unique elements
    
    Assumes that kurtosis tensor elements are arranged as follows:
    
        K0000 K1111 K2222 K0001 K0002
        K0111 K1112 K0222 K1222 K0011
        K0022 K1122 K0012 K0112 K0122 )    
    
    Parameters
    ----------- 

    kunique : list of unique kurtosis tensor elements

    
    Returns
    ----------- 

    K  : Full 4x4 kurtosis tensor


    """
    K=np.zeros((3,3,3,3))
    indices=np.array([1,16,81,2,3,8,24,27,54,4,9,36,6,12,18])
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for l in range(3):
                    K[i,j,k,l]=kunique[indices==(i+1)*(j+1)*(k+1)*(l+1)]
    return K

def SingleTensorKurtosis(bvals,gradients,S0,dunique,kunique,snr=None):
    """ Simulated signal with a single kurtosis tensor
     
    Parameters
    ----------- 
    bvals : array, shape (N,)

    gradients : array, shape (N,3) also known as bvecs

    S0 : double,

    dunique: array, shape (1,6) of unique diffusion tensor elements

    kunique: array, shape (1,15) of unique diffusion kurtosis tensor elements   

    snr : signal to noise ratio assuming gaussian noise. 
        Provide None for no noise.

    
    Returns
    --------
    Sk : simulated signal    
    
    """
    Sk =np.zeros(len(gradients))
    #D=np.dot(np.dot(evecs,np.diag(evals)),evecs.T) # diffusion tensor
    #dunique = [D[0,0],D[1,1],D[2,2],D[0,1],D[0,2],D[1,2]] # unique elements of diffusion tenosr
    MDsq = np.average(dunique[0],dunique[1],dunique[2])**2 # mean diffusivity 
    
    Ad = np.zeros([len(gradients),6])
    Ak = np.zeros([len(gradients),15])
    for (i,g) in enumerate(gradients):
        b = bvals[i]
        g0,g1,g2=g
        
        Ad[i] = [b*g0**2,b*g1**2,b*g2**2,
                 2*b*g0*g1,2*b*g0*g2,2*b*g1*g2]
        
        Ak[i] = [b*b*g0**4,
                 b*b*g1**4,
                 b*b*g2**4,
                 4*b*b*g0**3*g1,
                 4*b*b*g0**3*g2,
                 4*b*b*g1**3*g0,
                 4*b*b*g1**3*g2,
                 4*b*b*g2**3*g0,
                 4*b*b*g2**3*g1,
                 6*b*b*g0**2*g1**2,
                 6*b*b*g0**2*g2**2,
                 6*b*b*g1**2*g2**2,
                 12*b*b*g0**2*g1*g2,
                 12*b*b*g1**2*g0*g2,
                 12*b*b*g2**2*g0*g1]
                
        
        ld = Ad[i]*dunique #ld = np.dot(np.dot(g.T,D),g)
        lk = 1/6.*MDsq*Ak[i]*kunique        
        #S[i]=S0*np.exp(-bvals[i]*np.dot(np.dot(g.T,D),g)) # simulated signal in SingleTensor function above
        Sk[i]=S0*np.exp(-np.sum(ld)+np.sum(lk))
    
    if snr!=None:
        std=S0/snr
        Sk=Sk+np.random.randn(len(Sk))*std
        """ Add Rician noise option here? """        
    return Sk

