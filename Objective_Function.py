import numpy as np
import cv2 as cv
from GLCM import GLCM
from GLRM import GLRM
from Global_Vars import Global_Vars
from LBP import LBP
from LGP import get_lgp_descriptor
from Model_DeepLab import Train_Deeplab

def Objfun_Model1(Soln):
    Images = Global_Vars.Images
    Targets = Global_Vars.Target
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        Train_Data = Images
        Train_Target = Targets
        Train_Data = np.asarray(Train_Data)
        Train_Target = np.asarray(Train_Target)
        Deeplab_Im = Train_Deeplab(Train_Data, Train_Target, Train_Data,sol.astype('int'))
        correlation = np.zeros((len(Deeplab_Im)))
        for k in range(len(Deeplab_Im)):
            correlation[k] = np.corrcoef(Deeplab_Im[k])
        corr_coeff = np.mean(correlation)
        Variance = np.var(Deeplab_Im)
        Fitn[i] = 1/(corr_coeff+Variance)
    return Fitn

def Objfun_Model2(Soln):
    Images = Global_Vars.Images
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        GLCMs = []
        GLRMs=[]
        for i in range(len(Images)):
            Image1 = GLCM(Images[i], sol)
            Image2 = GLRM(Images[i], sol)
            GLCMs.append(Image1.astype('uint8'))
            GLRMs.append(Image2.astype('uint8'))
        Feat = cv.add(np.asarray(GLCMs), np.asarray(GLRMs))
        correlation = np.zeros((len(Feat)))
        for k in range(len(Feat)):
            correlation[k] = np.corrcoef(Feat[k])
        corr_coeff = np.mean(correlation)
        Variance = np.var(Feat)
        Fitn[i] = 1/(corr_coeff+Variance)
    return Fitn

def Objfun_Model3(Soln):
    Images = Global_Vars.Images
    if Soln.ndim == 2:
        v = Soln.shape[0]
        Fitn = np.zeros((Soln.shape[0], 1))
    else:
        v = 1
        Fitn = np.zeros((1, 1))
    for i in range(v):
        soln = np.array(Soln)

        if soln.ndim == 2:
            sol = Soln[i]
        else:
            sol = Soln
        imgs1 = []
        imgs2 = []
        for i in range(len(Images)):
            lgp = get_lgp_descriptor(Images[i].astype('uint8'),sol.astype('int'))
            lbp = LBP(Images[i],sol.astype('int'))
            imgs1.append(lgp)
            imgs2.append(lbp)
        Feat = cv.add(np.asarray(imgs1), np.asarray(imgs2))
        correlation = np.zeros((len(Feat)))
        for k in range(len(Feat)):
            correlation[k] = np.corrcoef(Feat[k])
        corr_coeff = np.mean(correlation)
        Variance = np.var(Feat)
        Fitn[i] = 1/(corr_coeff+Variance)
    return Fitn

