import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from sklearn.feature_extraction.image import reconstruct_from_patches_2d


def get_lgp_descriptor(img, sol):
    r=sol[0].astype('int')
    window_shape = (50,50)                      # size of neighbours
    B = extract_patches_2d(img, window_shape)   # 50*50 patches
    codes = np.zeros((B.shape[0], 50, 50))
    for i in range(0, B.shape[0]):              # for all patches find LGP
        padded = np.pad(B[i], (r, r), 'constant')
        a1 = padded[:-2*r, :-2*r]
        b1 = padded[:-2*r, r:-r]
        a2 = padded[:-2*r, 2*r:]
        b2 = padded[r:-r, 2*r:]
        a3 = padded[2*r:, 2*r:]
        b3 = padded[2*r:, r:-r]
        a4 = padded[2*r:, :-2*r]
        b4 = padded[r:-r, :-2*r]
        codes[i][:][:] = (a1 >= a3) + 2*(a2 >= a4) + 4*(b1 >= b3) + 8*(b2 >= b4)
    image = reconstruct_from_patches_2d(codes, img.shape)  # concatenate all patches into a single image
    return image.astype('uint8')     # return LGP image