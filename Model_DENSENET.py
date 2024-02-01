import numpy as np
from keras import backend as K
from keras.layers import *
from keras.models import *
from Evaluation import evaluation

def call(self, x):
    e = K.tanh(K.dot(x,self.W)+self.b)
    a = K.softmax(e, axis=1)
    output = x*a
    if self.return_sequences:

        return output
    return K.sum(output, axis=1)

def build(self, input_shape):
    self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1),
                             initializer="normal")
    self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1),
                             initializer="zeros")

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        a = K.softmax(e, axis=1)
        output = x

def conv_layer(conv_x, filters):
    conv_x = BatchNormalization()(conv_x)
    conv_x = Activation('relu')(conv_x)
    conv_x = Conv2D(filters, (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(conv_x)
    conv_x = Dropout(0.2)(conv_x)

    return conv_x


def dense_block(block_x, filters, growth_rate, layers_in_block):
    for i in range(layers_in_block):
        each_layer = conv_layer(block_x, growth_rate)
        block_x = concatenate([block_x, each_layer], axis=-1)
        filters += growth_rate

    return block_x, filters


def transition_block(trans_x, tran_filters):
    trans_x = BatchNormalization()(trans_x)
    trans_x = Activation('relu')(trans_x)
    trans_x = Conv2D(tran_filters, (1, 1), kernel_initializer='he_uniform', padding='same', use_bias=False)(trans_x)
    trans_x = AveragePooling2D((2, 2), strides=(2, 2))(trans_x)

    return trans_x, tran_filters

def HighRanking(Pred):
    Pred = np.asarray(Pred)
    pred = np.zeros((Pred.shape[1], 1))
    for i in range(Pred.shape[1]):
        p = Pred[:, i]
        uniq, count = np.unique(p, return_counts=True)
        index = np.argmax(count)
        pred[i] = uniq[index]
    return pred

def dense_net(sol,num_of_class=None):
    dense_block_size = 3
    layers_in_block = 4
    growth_rate = 12
    filters = growth_rate * 2
    input_img = Input(shape=(32, 32, 3))
    x = Conv2D(sol[0], (3, 3), kernel_initializer='he_uniform', padding='same', use_bias=False)(input_img)

    dense_x = Activation('relu')(x)
    dense_x = MaxPooling2D((3, 3), strides=(2, 2), padding='same')(dense_x)
    for block in range(dense_block_size - 1):
        dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
        dense_x, filters = transition_block(dense_x, filters)
    dense_x, filters = dense_block(dense_x, filters, growth_rate, layers_in_block)
    dense_x = BatchNormalization()(dense_x)
    dense_x = Activation('relu')(dense_x)
    dense_x = GlobalAveragePooling2D()(dense_x)
    output = Dense(num_of_class, activation='softmax')(dense_x)
    model = Model(input_img, output)
    return model

def Model_DENSENET_Highranking(train_data,train_target,test_data, test_target,sol):
    IMG_SIZE = [32, 32, 3]
    Data1 = np.zeros((train_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    Data2 = np.zeros((test_data.shape[0], IMG_SIZE[0], IMG_SIZE[1] * IMG_SIZE[2]))
    for i in range(train_data.shape[0]):
        Data1[i, :, :] = np.resize(train_data[i], (IMG_SIZE[1] , IMG_SIZE[2]*IMG_SIZE[0]))
    Datas = Data1.reshape(Data1.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    for i in range(test_data.shape[0]):
        Data2[i, :, :] = np.resize(test_data[i], (IMG_SIZE[1] , IMG_SIZE[2]*IMG_SIZE[0]))
    Data = Data2.reshape(Data2.shape[0], IMG_SIZE[0], IMG_SIZE[1], IMG_SIZE[2])
    model = dense_net(sol,num_of_class=train_target.shape[1])
    model.summary()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Datas, train_target.astype('float'), steps_per_epoch=1, epochs=sol[1])
    pred = model.predict(Data)
    return pred

def Model_DENSENET(Feat1,Feat2,Feat3,Target,sol=None):
    if sol is None:
        sol = [5,5]
    learnper = round(Feat1.shape[0] * 0.75)
    train_data1 = Feat1[:learnper, :]
    train_target1 = Target[:learnper, :]
    test_data1 = Feat1[learnper:, :]
    test_target1 = Target[learnper:, :]
    High_Ranking_Model1 = Model_DENSENET_Highranking(train_data1, train_target1, test_data1, test_target1, sol)
    train_data2 = Feat2[:learnper, :]
    train_target2 = Target[:learnper, :]
    test_data2 = Feat2[learnper:, :]
    test_target2 = Target[learnper:, :]
    High_Ranking_Model2 = Model_DENSENET_Highranking(train_data2, train_target2, test_data2, test_target2, sol)
    train_data3 = Feat3[:learnper, :]
    train_target3 = Target[:learnper, :]
    test_data3 = Feat3[learnper:, :]
    test_target3 = Target[learnper:, :]
    High_Ranking_Model3 = Model_DENSENET_Highranking(train_data3, train_target3, test_data3, test_target3, sol)
    high_ranking = [High_Ranking_Model1, High_Ranking_Model2, High_Ranking_Model3]
    predict = HighRanking(high_ranking)
    Eval = evaluation(predict, test_target3)
    return Eval