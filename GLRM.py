import numpy as np
from skimage.feature import greycomatrix

def GLRM(image,sol):
    # Specify the distances and angles for GLRLM
    distances = [1, 2, sol[0].astype('int')]
    angles = [0, np.pi/4, sol[1], 3*np.pi/4]
    # Compute GLRLM
    glrlm = greycomatrix(image.astype('uint8'), distances=distances, angles=angles, symmetric=True, normed=True)
    return glrlm


