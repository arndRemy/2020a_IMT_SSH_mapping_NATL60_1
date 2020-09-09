import scipy 
import numpy as np

def hanning2d(M, N):
    """
    A 2D hanning window, as per IDL's hanning function.  See numpy.hanning for the 1d description
    """
    
    if N <= 1:
        return np.hanning(M)
    elif M <= 1:
        return np.hanning(N) # scalar unity; don't window if dims are too small
    else:
        return np.outer(np.hanning(M),np.hanning(N))

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(phi, rho)

def Imputing_NaN(data, invalid=None):
    """
    Replace the value of invalid 'data' cells (indicated by 'invalid') 
    by the value of the nearest valid data cell
    """
    import scipy.ndimage as nd
    if invalid is None: invalid = np.isnan(data)
    ind = nd.distance_transform_edt(invalid, return_distances=False, return_indices=True)
    return data[tuple(ind)]

def raPsd2dv1(img,res,hanning):
    """ Computes and plots radially averaged power spectral density (power
     spectrum) of image IMG with spatial resolution RES.
    """
    img = img.copy()
    N, M = img.shape
    if hanning:
        img = hanning2d(*img.shape) * img       
    img =  Imputing_NaN(img)     
    imgf = np.fft.fftshift(np.fft.fft2(img))
    imgfp = np.power(np.abs(imgf)/(N*M),2)    
    # Adjust PSD size
    dimDiff = np.abs(N-M)
    dimMax = max(N,M)
    if (N>M):
        if ((dimDiff%2)==0):
            imgfp = np.pad(imgfp,((0,0),(int(dimDiff/2),int(dimDiff/2))),'constant',constant_values=np.nan)
        else:
            imgfp = np.pad(imgfp,((0,0),(int(dimDiff/2),1+int(dimDiff/2))),'constant',constant_values=np.nan)
            
    elif (N<M):
        if ((dimDiff%2)==0):
            imgfp = np.pad(imgfp,((int(dimDiff/2),int(dimDiff/2)),(0,0)),'constant',constant_values=np.nan)
        else:
            imgfp = np.pad(imgfp,((int(dimDiff/2),1+int(dimDiff/2)),(0,0)),'constant',constant_values=np.nan)
    halfDim = int(np.ceil(dimMax/2.))
    X, Y = np.meshgrid(np.arange(-dimMax/2.,dimMax/2.-1+0.00001),np.arange(-dimMax/2.,dimMax/2.-1+0.00001))           
    theta, rho = cart2pol(X, Y)                                              
    rho = np.round(rho+0.5)   
    Pf = np.zeros(halfDim)
    f1 = np.zeros(halfDim)
    for r in range(halfDim):
      Pf[r] = np.nansum(imgfp[rho == (r+1)])
      f1[r] = float(r+1)/dimMax
    f1 = f1/res
    return f1, Pf

def err_raPsd2dv1(img,imgref,res,hanning):
    """ Computes and plots radially averaged power spectral density error (power
     spectrum).
    """
    f_, Pf_ = raPsd2dv1(img-imgref,res,hanning)
    Pf_     = (Pf_/raPsd2dv1(imgref,res,hanning)[1])
    return f_, Pf_

def avg_raPsd2dv1(img3d,res,hanning):
    """ Computes and plots radially averaged power spectral density mean (power
     spectrum) of an image set img3d along the first dimension.
    """
    N = img3d.shape[0]
    for i in range(N):
        img=img3d[i,:,:]
        f_, Pf_ = raPsd2dv1(img,res,hanning)
        if i==0:
            f, Pf = f_, Pf_
        else:
            f = np.vstack((f,f_))
            Pf= np.vstack((Pf,Pf_))
    Pf = np.mean(Pf,axis=0)
    return f_, Pf

def avg_err_raPsd2dv1(img3d,img3dref,res,hanning):
    """ Computes and plots radially averaged power spectral density error mean (power
     spectrum) of an image set img3d along the first dimension.
    """
    N = img3d.shape[0]
    for i in range(N):
        img1 = img3d[i,:,:]
        img2 = img3dref[i,:,:]
        f_, Pf_ = raPsd2dv1(img1-img2,res,hanning)
        Pf_     = (Pf_/raPsd2dv1(img2,res,hanning)[1])
        if i==0:
            f, Pf = f_, Pf_
        else:
            f = np.vstack((f,f_))
            Pf= np.vstack((Pf,Pf_))
    Pf = np.mean(Pf,axis=0)
    return f_, Pf


