import numpy as np
from skimage.feature import greycomatrix, greycoprops


def GLCM(image,sol):
    # Specify the distances and angles for GLCM
    distances = [1, 2, sol[0].astype('int')]
    angles = [0, np.pi / 4, sol[1], 3 * np.pi / 4]
    # Specify the number of grey levels (typically 256 for an 8-bit image)
    levels = sol[2].astype('int')
    # Compute GLCM
    glcm = greycomatrix(image.astype('uint8'), distances=distances, angles=angles, symmetric=True, normed=True, levels=levels)
    return glcm

