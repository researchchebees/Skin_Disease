import numpy as np
import cv2 as cv
from keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D
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

def Model_RESNET_Highranking(train_data, train_target, test_data, test_target, sol=None):
    if sol is None:
        sol = [1]
    activation = ['Relu', 'linear', 'tanh', 'sigmoid']
    optimizer = ['SGD', 'RMSprop', 'Adam', 'Adadelta', 'Adagrad']
    IMG_SIZE = [224, 224, 3]
    Feat1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Feat1[i, :] = cv.resize(train_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    train_data = Feat1.reshape(Feat1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Feat2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(test_data.shape[0]):
        Feat2[i, :] = cv.resize(test_data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    test_data = Feat2.reshape(Feat2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])


    base_model = Sequential()
    base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
    act = ['Relu', 'linear', 'tanh', 'sigmoid']
    base_model.add(Dense(units=train_target.shape[1], activation=act[sol[0]]))
    base_model.compile(loss='binary_crossentropy', metrics=['acc'])
    try:
        base_model.fit(train_data, train_target,epochs=sol[1])
    except:
         pred = np.round(base_model.predict(test_data)).astype('int')
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return pred

def Model_RESNET(Feat1,Feat2,Feat3,Target,sol=None):
    if sol is None:
        sol = [5,5]
    learnper = round(Feat1.shape[0] * 0.75)
    train_data1 = Feat1[:learnper, :]
    train_target1 = Target[:learnper, :]
    test_data1 = Feat1[learnper:, :]
    test_target1 = Target[learnper:, :]
    High_Ranking_Model1 = Model_RESNET_Highranking(train_data1, train_target1, test_data1, test_target1, sol)
    train_data2 = Feat2[:learnper, :]
    train_target2 = Target[:learnper, :]
    test_data2 = Feat2[learnper:, :]
    test_target2 = Target[learnper:, :]
    High_Ranking_Model2 = Model_RESNET_Highranking(train_data2, train_target2, test_data2, test_target2, sol)
    train_data3 = Feat3[:learnper, :]
    train_target3 = Target[:learnper, :]
    test_data3 = Feat3[learnper:, :]
    test_target3 = Target[learnper:, :]
    High_Ranking_Model3 = Model_RESNET_Highranking(train_data3, train_target3, test_data3, test_target3, sol)
    high_ranking = [High_Ranking_Model1, High_Ranking_Model2, High_Ranking_Model3]
    predict = HighRanking(high_ranking)
    Eval = evaluation(predict, test_target3)
    return Eval