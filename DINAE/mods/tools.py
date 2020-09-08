from DINAE import *

# function to create recursive paths
def mk_dir_recursive(dir_path):
    if os.path.isdir(dir_path):
        return
    h, t = os.path.split(dir_path)  # head/tail
    if not os.path.isdir(h):
        mk_dir_recursive(h)

    new_path = join_paths(h, t)
    if not os.path.isdir(new_path):
        os.mkdir(new_path)

def Gradient(img, order):
    """ calculate x, y gradient and magnitude """ 
    sobelx = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=3)
    sobelx = sobelx/8.0
    sobely = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=3)
    sobely = sobely/8.0
    sobel_norm = np.sqrt(sobelx*sobelx+sobely*sobely)
    if (order==0):
        return sobelx
    elif (order==1):
        return sobely
    else:
        return sobel_norm

def insert_Sobel(size_tw,dir="x"):
    kernel_weights=np.zeros((3,3,size_tw,size_tw))
    if dir=="x":
        sobel=np.array([[-1,0,1],[-2,0,2],[-1,0,1]]).T
    if dir=="y":
        sobel=np.array([[-1,-2,-1],[0,0,0],[1,2,1]]).T
    for i in range(size_tw):
        kernel_weights[:,:,i,i]=sobel
    return kernel_weights

def thresholding(x,thr):
    greater = K.greater_equal(x,thr) #will return boolean values
    greater = K.cast(greater, dtype=K.floatx()) #will convert bool to 0 and 1    
    return greater

def ifelse(cond1,val1,val2):
    if cond1==True:
        res = val1
    else:
        res = val2
    return res

def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")


