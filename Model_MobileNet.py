import cv2 as cv
import numpy as np
from PIL import Image
from keras.applications import MobileNet
from keras.applications.inception_resnet_v2 import preprocess_input
from keras_preprocessing.image import img_to_array
from Evaluation import evaluation

def HighRanking(Pred):
    Pred = np.asarray(Pred)
    pred = np.zeros((Pred.shape[1], 1))
    for i in range(Pred.shape[1]):
        p = Pred[:, i]
        uniq, count = np.unique(p, return_counts=True)
        index = np.argmax(count)
        pred[i] = uniq[index]
    return pred

def Model_MobileNet_Highranking(Data, Target, test_data, test_tar, sol=None):
    if sol is None:
        sol = [50,50]
    model = MobileNet(classes=1)
    IMG_SIZE = [224, 224, 3]
    Feat = np.zeros((Target.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Data.shape[0]):
        Feat[i, :] = cv.resize(Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Data = Feat.reshape(Feat.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    for i in range(Data.shape[0]):
        data = Image.fromarray(np.uint8(Data[i])).convert('RGB')
        data = img_to_array(data)
        data = np.expand_dims(data, axis=0)
        data = np.squeeze(data)
        Data[i] = cv.resize(data, (224, 224))
        Data[i] = preprocess_input(Data[i])

    model.compile(optimizer='Adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])  # optimizer='Adam'

    model.fit(Data,Target.astype('float'),steps_per_epoch=sol[0],epochs=sol[1])
    preds = model.predict(test_data)
    return preds

def Model_MobileNet(Feat1,Feat2,Feat3,Target,sol=None):
    if sol is None:
        sol = [5,5]
    learnper = round(Feat1.shape[0] * 0.75)
    train_data1 = Feat1[:learnper, :]
    train_target1 = Target[:learnper, :]
    test_data1 = Feat1[learnper:, :]
    test_target1 = Target[learnper:, :]
    High_Ranking_Model1 = Model_MobileNet_Highranking(train_data1, train_target1, test_data1, test_target1, sol)
    train_data2 = Feat2[:learnper, :]
    train_target2 = Target[:learnper, :]
    test_data2 = Feat2[learnper:, :]
    test_target2 = Target[learnper:, :]
    High_Ranking_Model2 = Model_MobileNet_Highranking(train_data2, train_target2, test_data2, test_target2, sol)
    train_data3 = Feat3[:learnper, :]
    train_target3 = Target[:learnper, :]
    test_data3 = Feat3[learnper:, :]
    test_target3 = Target[learnper:, :]
    High_Ranking_Model3 = Model_MobileNet_Highranking(train_data3, train_target3, test_data3, test_target3, sol)
    high_ranking = [High_Ranking_Model1, High_Ranking_Model2, High_Ranking_Model3]
    predict = HighRanking(high_ranking)
    Eval = evaluation(predict, test_target3)
    return Eval