import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.models import load_model
from stats_graph import stats_graph
# 指定第一块GPU可用 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

KTF.set_session(sess)
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#tf.compat.v1.disable_eager_execution()
#config = tf.compat.v1.ConfigProto()
#config.gpu_options.allow_growth=True   #不全部占满显存, 按需分配
#sess = tf.compat.v1.Session(config=config)

from keras.utils import plot_model
import data_helper
from keras.layers import Embedding, Input, Bidirectional, LSTM, Concatenate, Add, Dropout, Dense, \
    BatchNormalization, Lambda, Activation, multiply, concatenate, Flatten, add, Dot,Permute
from keras.models import Model
import keras.backend as K
from keras.callbacks import *
from tensorflow.python.ops.nn import softmax

input_dim = data_helper.MAX_SEQUENCE_LENGTH
EMBDIM = data_helper.EMBDIM
embedding_matrix = data_helper.load_pickle('embedding_matrix.pkl')
model_data = data_helper.load_pickle('model_data.pkl')
embedding_layer = Embedding(embedding_matrix.shape[0], EMBDIM, weights = [embedding_matrix], trainable=False)


def align(input_1, input_2):
    attention = Dot(axes=-1)([input_1, input_2])
    w_att_1 = Lambda(lambda x: softmax(x, axis=1))(attention)
    w_att_2 = Permute((2, 1))(Lambda(lambda x: softmax(x, axis=2))(attention))
    in1_aligned = Dot(axes=1)([w_att_1, input_1])
    in2_aligned = Dot(axes=1)([w_att_2, input_2])

    in1_aligned = add([in1_aligned, input_1])
    in2_aligned = add([in2_aligned, input_2])

    return in1_aligned, in2_aligned

def base_model(input_shape):
    w_input = Input(shape = input_shape)
    c_input = Input(shape = input_shape)
    w_embedding = embedding_layer(w_input)
    c_embedding = embedding_layer(c_input)
    w_l = LSTM(300, return_sequences = True, dropout=0.5)(w_embedding)
    c_l = LSTM(300, return_sequences = True, dropout=0.5)(c_embedding)
    # w_align,w_lalign = align(w_embedding,w_l)
    # c_align, c_lalign = align(c_embedding, c_l)
    w = Concatenate()([w_embedding, w_l])
    c = Concatenate()([c_embedding, c_l])
    
   # p = concatenate([w,c])


    model = Model([w_input, c_input],[w, c], name = 'base_model')
    model.summary()
    return model

def f1_score(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def recall(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    recall = c1 / c3

    return recall

def precision(y_true, y_pred):

    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    return precision

margin = 0.6
theta = lambda t: (K.sign(t)+1.)/2.

def loss(y_true, y_pred):
    return -(1 - theta(y_true - margin) * theta(y_pred - margin) - theta(1 - margin - y_true) * theta(1-margin-y_pred)) * (y_true*K.log(y_pred + 1e-8) + (1-y_true)*K.log(1-y_pred+1e-8))

def matching(p,q):
    abs_diff = Lambda(lambda x: K.abs(x[0] - x[1]))([p, q])
    #cos_diff = Lambda(lambda x: K.cos(x[0] - x[1]))([p, q])
    multi_diff = multiply([p, q])
    all_diff = concatenate([abs_diff, multi_diff])
    return all_diff

def siamese_model():
    input_shape = (input_dim,)
    input_p1 = Input(shape = input_shape)
    input_p2 = Input(shape = input_shape)
    input_p3 = Input(shape = input_shape)
    input_p4 = Input(shape = input_shape)
    base_net = base_model(input_shape)

    pw, pc = base_net([input_p1, input_p3])
    qw, qc = base_net([input_p2, input_p4])
    qw_align, pw_align = align(qw, pw)
    pc_align, qc_align = align(pc, qc)
    p_align = Add()([pw_align, pc_align])
    q_align = Add()([qw_align, qc_align])

    
    pl = LSTM(300, return_sequences = True, dropout = 0.5)(p_align)
    ql = LSTM(300, return_sequences = True, dropout = 0.5)(q_align)

    
    # doing matching
   # all_diff1 = matching(pl,ql)
    p1 = Bidirectional(LSTM(300, return_sequences = True, dropout = 0.5), merge_mode = 'sum')(Concatenate()([p_align,pl]))
    q1 = Bidirectional(LSTM(300, return_sequences = True, dropout = 0.5), merge_mode = 'sum')(Concatenate()([q_align,ql]))

    
    
    all_diff = matching(p1,q1)

    


    # print(all_diff)



    # DNN
    all_diff = Dropout(0.5)(all_diff)

    similarity = Dense(600)(all_diff)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('relu')(similarity)
    similarity = Flatten()(similarity)
    similarity = Dense(600)(similarity)
    similarity = Dropout(0.5)(similarity)
    similarity = Activation('relu')(similarity)
    #
    similarity = Dense(1)(similarity)
    similarity = BatchNormalization()(similarity)
    similarity = Activation('sigmoid')(similarity)
    model = Model([input_p1, input_p2, input_p3, input_p4], [similarity])
    # loss:binary_crossentropy;optimizer:adm,Adadelta
    model.summary()



#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy', precision, recall, f1_score])

    return model

    
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', precision, recall, f1_score])
    #model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    #model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics=['acc'])
    #return model


def train():
    data = data_helper.load_pickle('model_data.pkl')

    train_q1 = data['train_q1']
    train_q2 = data['train_q2']
    train_q3 = data['train_q3']
    train_q4 = data['train_q4']
    train_y = data['train_label']

    dev_q1 = data['dev_q1']
    dev_q2 = data['dev_q2']
    dev_q3 = data['dev_q3']
    dev_q4 = data['dev_q4']
    dev_y = data['dev_label']

    test_q1 = data['test_q1']
    test_q2 = data['test_q2']
    test_q3 = data['test_q3']
    test_q4 = data['test_q4']
    test_y = data['test_label']
    model_path = 'weights.best.hdf5'
    #tensorboard_path = 'tensorboard'
    model = siamese_model()
    sess = K.get_session()
    graph = sess.graph
    stats_graph(graph)
    checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', period=1)
    #tensorboard = TensorBoard(log_dir=tensorboard_path)
    earlystopping = EarlyStopping(monitor='val_acc', patience=10, verbose=0, mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='val_acc', patience=5, mode='max')
    callbackslist = [checkpoint, earlystopping, reduce_lr]

    history = model.fit([train_q1, train_q2, train_q3, train_q4], train_y,
                        batch_size=512,
                        epochs=200,
                        validation_data=([dev_q1, dev_q2, dev_q3, dev_q4], dev_y),
                        callbacks=callbackslist)
    '''
    ## Add graphs here
    import matplotlib.pyplot as plt

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])   
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['precision'])
    plt.plot(history.history['val_precision'])
    plt.plot(history.history['recall'])
    plt.plot(history.history['val_recall'])
    plt.plot(history.history['f1_score'])
    plt.plot(history.history['val_f1_score'])
    plt.xlabel('epoch')
    plt.legend(['train loss', 'val loss','train accuracy', 'val accuracy','train precision', 'val precision','train recall', 'val recall','train f1_score', 'val f1_score'], loc=3,
               bbox_to_anchor=(1.05,0),borderaxespad=0)
    pic = plt.gcf()
    pic.savefig ('pic.eps',format = 'eps',dpi=1000)
    plt.show()
    '''
    loss, accuracy, precision, recall, f1_score = model.evaluate([test_q1, test_q2, test_q3, test_q4], test_y, verbose=1, batch_size=256)
    print("Test best model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (
    loss, accuracy, precision, recall, f1_score))
    x = "Test best model =loss: %.4f, accuracy:%.4f, precision:%.4f,recall: %.4f, f1_score:%.4f" % (loss, accuracy, precision, recall, f1_score)+'\n'
    model.save(model_path)
    with open('record.txt','a') as f:
      f.write(x)


      


if __name__ == '__main__':
    train()




