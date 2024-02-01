import numpy as np
import cv2 as cv
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def Image_Results():
    Orig = np.load('Org_Img.npy',allow_pickle=True)
    Images = np.load('Pre_Processed_Images.npy', allow_pickle=True)
    segment = np.load('Deeplab_Images.npy', allow_pickle=True)
    grnd = np.load('Ground_Truth.npy', allow_pickle=True)
    ind = [0,1,2,3,4]
    for j in range(len(ind)):
        original = Orig[ind[j]]
        image = Images[ind[j]].astype('uint8')
        seg = segment[ind[j]]
        Output = np.zeros((seg.shape)).astype('uint8')
        max= np.max(seg)
        ind1 = np.where(seg > 0.1)
        Output[ind1] = 255
        # cv.imshow('im', Output)
        # cv.waitKey(0)
        gt = grnd[ind[j]]
        fig, ax = plt.subplots(1, 4)
        plt.suptitle("Image %d" % (j + 1), fontsize=20)
        plt.subplot(1, 4, 1)
        plt.title('Orig')
        plt.imshow(original)
        plt.subplot(1, 4, 2)
        plt.title('Prep')
        plt.imshow(image)
        plt.subplot(1, 4, 3)
        plt.title('Seg')
        plt.imshow(Output)
        plt.subplot(1, 4, 4)
        plt.title('GT')
        plt.imshow(gt)
        path1 = "./Results/Images/Dataset_%simage.png" % (j + 1)
        plt.savefig(path1)
        plt.show()
if __name__ == '__main__':
    Image_Results()