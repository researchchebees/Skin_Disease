import os
import numpy as np
import cv2 as cv
import pandas as pd
from numpy import matlib
import random as rn
from GLCM import GLCM
from GLRM import GLRM
from GWO import GWO
from Global_Vars import Global_Vars
from Image_Resuts import Image_Results
from LBP import LBP
from LGP import get_lgp_descriptor
from Model_DENSENET import Model_DENSENET
from Model_DeepLab import Train_Deeplab
from Model_ENSEMBLE import Model_MMCS_RDA_EHR_TM
from Model_INCEPTION import Model_INCEPTION
from Model_MobileNet import Model_MobileNet
from Model_Resnet import Model_RESNET
from Model_VGG16 import Model_VGG16
from Objective_Function import Objfun_Model1, Objfun_Model2, Objfun_Model3
from PROPOSED import PROPOSED
from PSO import PSO
from Plot_Results import plot_results_learnperc, Statistical_, plot_results_kfold, plot_Comp
from RDA import RDA
from WOA import WOA

# Read Dataset
an = 0
if an == 1:
    directory = './PH2Dataset/PH2 Dataset images/'
    tar_directory = './PH2Dataset/PH2_dataset.xlsx'
    dir = os.listdir(directory)
    images = []
    for i in range(len(dir)):
        file = directory + dir[i] + '/'
        dir1 = os.listdir(file)
        for j in range(len(dir1)):
            file1 = file + dir1[j] + '/'
            if 'Dermoscopic' in dir1[j]:
                dir2 = os.listdir(file1)
                for k in range(len(dir2)):
                    file2 = file1 + dir2[k]
                    read = cv.imread(file2)
                    read = cv.resize(read, [128, 128])
                    images.append(read)
    tar = pd.read_excel(tar_directory)
    tar = np.asarray(tar)
    target = tar[12:, 5]
    np.save('Org_Img.npy', np.asarray(images))
    np.save('Targets', target.reshape(-1, 1))

# Read Groundtruth
an = 0
if an == 1:
    directory = './PH2Dataset/PH2 Dataset images/'
    dir = os.listdir(directory)
    images = []
    for i in range(len(dir)):
        file = directory + dir[i] + '/'
        dir1 = os.listdir(file)
        for j in range(len(dir1)):
            file1 = file + dir1[j] + '/'
            if 'lesion' in dir1[j]:
                dir2 = os.listdir(file1)
                for k in range(len(dir2)):
                    file2 = file1 + dir2[k]
                    read = cv.imread(file2)
                    read = cv.resize(read, [128, 128])
                    gray = cv.cvtColor(read, cv.COLOR_RGB2GRAY)
                    images.append(gray)
    np.save('Ground_Truth.npy', np.asarray(images))

# Pre_Proceesing
an = 0
if an == 1:
    pre = []
    images = np.load('Org_Img.npy', allow_pickle=True)
    for i in range(len(images)):
        gray = cv.cvtColor(images[i], cv.COLOR_RGB2GRAY)
        # Apply Gaussian blur to the image
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        # Apply CLAHE to the image
        clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(blurred)
        clr = cv.cvtColor(enhanced, cv.COLOR_GRAY2RGB)
        pre.append(clr)
    np.save('Pre_Processed_Images.npy', np.asarray(pre))

# Optimization for Segmentation
an = 0
if an == 1:
    Images = np.load('Pre_Processed_Images.npy', allow_pickle=True)
    Targets = np.load('Ground_Truth.npy', allow_pickle=True)
    Global_Vars.Images = Images
    Global_Vars.Target = Targets
    Npop = 10
    Chlen = 2
    xmin = matlib.repmat([5, 50], Npop, 1)
    xmax = matlib.repmat([255, 100], Npop, 1)
    fname = Objfun_Model1
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.asarray(rn.uniform(xmin[p1, p2], xmax[p1, p2]))
    Max_iter = 50

    print("PSO...")
    [bestfit1, fitness1, bestsol1, time1] = PSO(initsol, fname, xmin, xmax, Max_iter)

    print("GWO...")
    [bestfit2, fitness2, bestsol2, time2] = GWO(initsol, fname, xmin, xmax, Max_iter)

    print("WOA...")
    [bestfit4, fitness4, bestsol4, time3] = WOA(initsol, fname, xmin, xmax, Max_iter)

    print("RDA...")
    [bestfit3, fitness3, bestsol3, time4] = RDA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    Bestsol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
    np.save('Bestsol_Model1.npy', Bestsol)

##  DeeplabV3 Segmentation
an = 0
if an == 1:
    Train_Data = np.load('Pre_Processed_Images.npy', allow_pickle=True)
    Train_Target = np.load('Ground_Truth.npy', allow_pickle=True)
    sol = np.load('Bestsol_Model1.npy', allow_pickle=True)[4, :]
    Train_Data = np.asarray(Train_Data)
    Train_Target = np.asarray(Train_Target)
    Deeplab_Im = Train_Deeplab(Train_Data, Train_Target, Train_Data, sol.astype('int'))
    np.save('Deeplab_Images.npy', Deeplab_Im)

# Optimization for GLCM and GLRM
an = 0
if an == 1:
    Images = np.load('Deeplab_Images.npy', allow_pickle=True)
    Global_Vars.Images = Images
    Npop = 10
    Chlen = 3
    xmin = matlib.repmat([1, 0, 250], Npop, 1)
    xmax = matlib.repmat([3, 3 * np.pi / 4, 500], Npop, 1)
    fname = Objfun_Model2
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.asarray(rn.uniform(xmin[p1, p2], xmax[p1, p2]))
    Max_iter = 50

    print("PSO...")
    [bestfit1, fitness1, bestsol1, time1] = PSO(initsol, fname, xmin, xmax, Max_iter)

    print("GWO...")
    [bestfit2, fitness2, bestsol2, time2] = GWO(initsol, fname, xmin, xmax, Max_iter)

    print("WOA...")
    [bestfit4, fitness4, bestsol4, time3] = WOA(initsol, fname, xmin, xmax, Max_iter)

    print("RDA...")
    [bestfit3, fitness3, bestsol3, time4] = RDA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    Bestsol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
    np.save('Bestsol_Model2.npy', Bestsol)

##  GLCM and GLRM
an = 0
if an == 1:
    Images = np.load('Deeplab_Images.npy', allow_pickle=True)
    sol = np.load('Bestsol_Model2.npy', allow_pickle=True)[4, :]
    GLRMs = []
    GLCMs = []
    for i in range(len(Images)):
        Image1 = GLCM(Images[i], sol)
        Image2 = GLRM(Images[i], sol)
        GLCMs.append(Image1.astype('uint8'))
        GLRMs.append(Image1.astype('uint8'))
    Feat = cv.add(np.asarray(GLCMs), np.asarray(GLRMs))
    np.save('GLCM_GLRM_Feat.npy', Feat)

# Optimization for LGP and LBP
an = 0
if an == 1:
    Images = np.load('Deeplab_Images.npy', allow_pickle=True)
    Global_Vars.Images = Images
    Npop = 10
    Chlen = 2
    xmin = matlib.repmat([1, 1], Npop, 1)
    xmax = matlib.repmat([5, 128], Npop, 1)
    fname = Objfun_Model3
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.asarray(rn.uniform(xmin[p1, p2], xmax[p1, p2]))
    Max_iter = 50

    print("PSO...")
    [bestfit1, fitness1, bestsol1, time1] = PSO(initsol, fname, xmin, xmax, Max_iter)

    print("GWO...")
    [bestfit2, fitness2, bestsol2, time2] = GWO(initsol, fname, xmin, xmax, Max_iter)

    print("WOA...")
    [bestfit4, fitness4, bestsol4, time3] = WOA(initsol, fname, xmin, xmax, Max_iter)

    print("RDA...")
    [bestfit3, fitness3, bestsol3, time4] = RDA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    Bestsol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
    np.save('Bestsol_Model3.npy', Bestsol)

# LGP and LBP Pattern Extraction
an = 0
if an == 1:
    imgs1 = []
    imgs2 = []
    Images = np.load('Deeplab_Images.npy', allow_pickle=True)
    sol = np.load('Bestsol_Model3.npy', allow_pickle=True)[4, :]
    for i in range(len(Images)):
        lgp = get_lgp_descriptor(Images[i].astype('uint8'), sol.astype('int'))
        lbp = LBP(Images[i], sol.astype('int'))
        imgs1.append(lgp)
        imgs2.append(lbp)
    Feat = cv.add(np.asarray(imgs1), np.asarray(imgs2))
    np.save('LGP_LBP_Feat.npy', Feat)

# Optimization for high Ranking Classification
an = 0
if an == 1:
    Feat1 = np.load('Deeplab_Images.npy', allow_pickle=True)
    Feat2 = np.load('GLCM_GLRM_Feat.npy', allow_pickle=True)
    Feat3 = np.load('LGP_LBP_Feat.npy', allow_pickle=True)
    Target = np.load('TargetS.npy', allow_pickle=True)
    Global_Vars.Feat1 = Feat1
    Global_Vars.Feat2 = Feat2
    Global_Vars.Feat3 = Feat3
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 10
    xmin = matlib.repmat([5, 50, 50, 50, 5, 50, 1, 50, 5, 50], Npop, 1)
    xmax = matlib.repmat([255, 100, 100, 100, 255, 100,4, 100, 255, 100], Npop, 1)
    fname = Objfun_Model3
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = np.asarray(rn.uniform(xmin[p1, p2], xmax[p1, p2]))
    Max_iter = 50

    print("PSO...")
    [bestfit1, fitness1, bestsol1, time1] = PSO(initsol, fname, xmin, xmax, Max_iter)

    print("GWO...")
    [bestfit2, fitness2, bestsol2, time2] = GWO(initsol, fname, xmin, xmax, Max_iter)

    print("WOA...")
    [bestfit4, fitness4, bestsol4, time3] = WOA(initsol, fname, xmin, xmax, Max_iter)

    print("RDA...")
    [bestfit3, fitness3, bestsol3, time4] = RDA(initsol, fname, xmin, xmax, Max_iter)

    print("PROPOSED...")
    [bestfit5, fitness5, bestsol5, time5] = PROPOSED(initsol, fname, xmin, xmax, Max_iter)

    Bestsol = ([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
    np.save('Bestsol.npy', Bestsol)

# ## Classification
an = 0
if an == 1:
    EVAL = []
    Learnper = [0.35, 0.55, 0.65, 0.75, 0.85]
    Eval_all = []
    Feat1 = np.load('Deeplab_Images.npy', allow_pickle=True)
    Feat2 = np.load('GLCM_GLRM_Feat.npy', allow_pickle=True)
    Feat3 = np.load('LGP_LBP_Feat.npy', allow_pickle=True)
    sol = np.load('Bestsol.npy', allow_pickle=True)
    Targets = np.load('TargetS.npy', allow_pickle=True)
    Global_Vars.Feat1 = Feat1
    Global_Vars.Feat2 = Feat2
    Global_Vars.Feat3 = Feat3
    Global_Vars.Target = Targets
    for i in range(len(Learnper)):
        Eval = np.zeros((11, 14))
        for j in range(sol.shape[0]):
            Eval = Model_MMCS_RDA_EHR_TM(Feat1, Feat2, Feat3, Targets, sol[j].astype('int'))
        Eval[5, :] = Model_DENSENET(Feat1, Feat2, Feat3, Targets)
        Eval[6, :] = Model_MobileNet(Feat1, Feat2, Feat3, Targets)
        Eval[7, :] = Model_VGG16(Feat1, Feat2, Feat3, Targets)
        Eval[8, :] = Model_RESNET(Feat1, Feat2, Feat3, Targets)
        Eval[9, :] = Model_INCEPTION(Feat1, Feat2, Feat3, Targets)
        Eval[10, :] = Eval[4, :]
        Eval_all.append(Eval)
        EVAL.append(Eval_all)
    np.save('Eval_all.npy', np.asarray(EVAL))

plot_results_learnperc()
plot_results_kfold()
plot_Comp()
Statistical_()
Image_Results()
