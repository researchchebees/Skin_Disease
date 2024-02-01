from Evaluation import evaluation
from Model_DENSENET import Model_DENSENET
import numpy as np
from Model_INCEPTION import Model_INCEPTION
from Model_MobileNet import Model_MobileNet
from Model_Resnet import Model_RESNET
from Model_VGG16 import Model_VGG16

def HighRanking(Pred):
    Pred = np.asarray(Pred)
    pred = np.zeros((Pred.shape[1], 1))
    for i in range(Pred.shape[1]):
        p = Pred[:, i]
        uniq, count = np.unique(p, return_counts=True)
        index = np.argmax(count)
        pred[i] = uniq[index]
    return pred

def Model_ENSEMBLE(Train_Data, Train_Target, Test_Data, Test_Target,sol):
    Eval1, pred1 = Model_DENSENET(Train_Data, Train_Target, Test_Data, Test_Target,sol[:2])
    Eval2, pred2 =  Model_MobileNet(Train_Data, Train_Target, Test_Data, Test_Target,sol[2:4])
    Eval3, pred3 = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target,sol[4:6])
    Eval4, pred4 = Model_RESNET(Train_Data, Train_Target, Test_Data, Test_Target,sol[6:8])
    Eval5, pred5 = Model_INCEPTION(Train_Data, Train_Target, Test_Data, Test_Target,sol[8:10])
    pred = [pred1, pred2, pred3,pred4,pred5]
    predict = HighRanking(pred)
    return predict

def Model_MMCS_RDA_EHR_TM(Feat1,Feat2,Feat3,Target,sol):
    learnper = round(Feat1.shape[0] * 0.75)
    train_data1 = Feat1[:learnper, :]
    train_target1 = Target[:learnper, :]
    test_data1 = Feat1[learnper:, :]
    test_target1 = Target[learnper:, :]
    High_Ranking_Model1 = Model_ENSEMBLE(train_data1,train_target1,test_data1,test_target1,sol)
    train_data2 = Feat2[:learnper, :]
    train_target2 = Target[:learnper, :]
    test_data2 = Feat2[learnper:, :]
    test_target2 = Target[learnper:, :]
    High_Ranking_Model2 = Model_ENSEMBLE(train_data2, train_target2, test_data2, test_target2, sol)
    train_data3 = Feat3[:learnper, :]
    train_target3 = Target[:learnper, :]
    test_data3 = Feat3[learnper:, :]
    test_target3 = Target[learnper:, :]
    High_Ranking_Model3 = Model_ENSEMBLE(train_data3, train_target3, test_data3, test_target3, sol)
    high_ranking = [High_Ranking_Model1, High_Ranking_Model2, High_Ranking_Model3]
    predict = HighRanking(high_ranking)
    Eval = evaluation(predict,test_target3)
    return Eval



