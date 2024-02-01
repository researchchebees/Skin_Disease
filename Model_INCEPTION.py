import numpy as np
from Evaluation import evaluation
from inception_v3 import InceptionV3
import cv2 as cv

def HighRanking(Pred):
    Pred = np.asarray(Pred)
    pred = np.zeros((Pred.shape[1], 1))
    for i in range(Pred.shape[1]):
        p = Pred[:, i]
        uniq, count = np.unique(p, return_counts=True)
        index = np.argmax(count)
        pred[i] = uniq[index]
    return pred

def Model_INCEPTION_Highranking(Train_Data, Train_Tar, Test_Data, Test_Tar, sol=None):
    if sol is None:
        sol = [5, 5]
    IMG_SIZE = [224, 224, 3]
    model = InceptionV3(weights='imagenet', include_top=True, epoch=sol[1], sol=sol[0])
    Feat = np.zeros((Train_Tar.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Feat[i, :] = cv.resize(Train_Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    print("model structure: ", model.summary())
    print("model weights: ", model.get_weights())
    Feat1 = np.zeros((Test_Data.shape[0], 1))
    for i in range(Test_Data.shape[0]):
        Feat1[i, :, ] = np.resize(Test_Data[i], 1)
    model.compile(loss='binary_crossentropy', metrics=['acc'])
    predict = np.round(model.predict(Feat1)).astype('int')
    return predict

def Model_INCEPTION(Feat1,Feat2,Feat3,Target,sol=None):
    if sol is None:
        sol = [5,5]
    learnper = round(Feat1.shape[0] * 0.75)
    train_data1 = Feat1[:learnper, :]
    train_target1 = Target[:learnper, :]
    test_data1 = Feat1[learnper:, :]
    test_target1 = Target[learnper:, :]
    High_Ranking_Model1 = Model_INCEPTION_Highranking(train_data1, train_target1, test_data1, test_target1, sol)
    train_data2 = Feat2[:learnper, :]
    train_target2 = Target[:learnper, :]
    test_data2 = Feat2[learnper:, :]
    test_target2 = Target[learnper:, :]
    High_Ranking_Model2 = Model_INCEPTION_Highranking(train_data2, train_target2, test_data2, test_target2, sol)
    train_data3 = Feat3[:learnper, :]
    train_target3 = Target[:learnper, :]
    test_data3 = Feat3[learnper:, :]
    test_target3 = Target[learnper:, :]
    High_Ranking_Model3 = Model_INCEPTION_Highranking(train_data3, train_target3, test_data3, test_target3, sol)
    high_ranking = [High_Ranking_Model1, High_Ranking_Model2, High_Ranking_Model3]
    predict = HighRanking(high_ranking)
    Eval = evaluation(predict, test_target3)
    return Eval