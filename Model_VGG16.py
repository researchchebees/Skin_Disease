from keras.layers import MaxPool2D
from keras.models import Sequential
from keras.layers.core import Flatten, Dense
from keras.layers.convolutional import *
import cv2 as cv, numpy as np
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

def VGG_16(sol,weights_path=None, num_of_class=None):
    model = Sequential()
    model.add(Conv2D(input_shape=(32, 32, 3), filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=sol[0], kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=4096, activation="relu"))
    model.add(Dense(units=num_of_class, activation="softmax"))
    return model

def Model_VGG16_Highranking(Train_Data, Train_Tar, Test_Data, Test_Tar,sol=None):
    if sol is None:
        sol = [5,5]
    ## VGG16
    IMG_SIZE = [32, 32, 3]
    Train1 = np.zeros((Train_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Train_Data.shape[0]):
        Train1[i, :, :] = cv.resize(Train_Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Train = Train1.reshape(Train1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])

    Test1 = np.zeros((Test_Data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(Test_Data.shape[0]):
        Test1[i, :, :] = cv.resize(Test_Data[i], (IMG_SIZE[1] * IMG_SIZE[2], IMG_SIZE[0]))
    Test = Test1.reshape(Test1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    model = VGG_16(sol,num_of_class=Train_Tar.shape[1])
    model.summary()
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.fit(x=Train, y=Train_Tar, epochs=sol[1], steps_per_epoch=1)
    predict = model.predict(Test).astype('int')
    return predict

def Model_VGG16(Feat1,Feat2,Feat3,Target,sol=None):
    if sol is None:
        sol = [5,5]
    learnper = round(Feat1.shape[0] * 0.75)
    train_data1 = Feat1[:learnper, :]
    train_target1 = Target[:learnper, :]
    test_data1 = Feat1[learnper:, :]
    test_target1 = Target[learnper:, :]
    High_Ranking_Model1 = Model_VGG16_Highranking(train_data1, train_target1, test_data1, test_target1, sol)
    train_data2 = Feat2[:learnper, :]
    train_target2 = Target[:learnper, :]
    test_data2 = Feat2[learnper:, :]
    test_target2 = Target[learnper:, :]
    High_Ranking_Model2 = Model_VGG16_Highranking(train_data2, train_target2, test_data2, test_target2, sol)
    train_data3 = Feat3[:learnper, :]
    train_target3 = Target[:learnper, :]
    test_data3 = Feat3[learnper:, :]
    test_target3 = Target[learnper:, :]
    High_Ranking_Model3 = Model_VGG16_Highranking(train_data3, train_target3, test_data3, test_target3, sol)
    high_ranking = [High_Ranking_Model1, High_Ranking_Model2, High_Ranking_Model3]
    predict = HighRanking(high_ranking)
    Eval = evaluation(predict, test_target3)
    return Eval