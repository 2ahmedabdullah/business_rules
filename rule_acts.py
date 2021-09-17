# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 15:47:05 2019

@author: 1449486
"""


import tensorflow.contrib.keras as keras
import tensorflow as tf

from keras.engine import Layer, InputSpec
from keras import regularizers, initializers, constraints
from keras import backend as K


class Attention(Layer):
      """
        Keras Layer that implements an Attention mechanism for temporal data.
        Supports Masking.
        Follows the work of Raffel et al. [https://arxiv.org/abs/1512.08756]
        # Input shape
            3D tensor with shape: `(samples, steps, features)`.
        # Output shape
            2D tensor with shape: `(samples, features)`.
        :param kwargs:
        Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
        The dimensions are inferred based on the output shape of the RNN.
        Example:
            model.add(LSTM(64, return_sequences=True))
            model.add(Attention())
        """
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

import os
import re
import string
import numpy as np
import pandas as pd

from os.path import expanduser as ospath

from keras.models import Sequential,Model
from keras.layers import Dense,Activation,LSTM,Dropout
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.layers import Bidirectional
from keras.layers import Input, dot, subtract, multiply,add,average
from keras.engine.topology import Layer

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import FastText
from keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import itertools
from pyexcel_ods import get_data

from keras.callbacks import EarlyStopping,ModelCheckpoint


from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import metrics


from keras.layers import Conv1D
from keras.layers import GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Concatenate, SpatialDropout1D
from keras.optimizers import Adam
from keras import backend as K
from keras import initializers, optimizers, layers
import matplotlib.pyplot as plt


def confusion_matrix_values(y_actual, y_pred):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for i in range(len(y_pred)): 
        if y_actual[i]==1:
            if y_pred[i]==1:
                TP = TP+1
            else:
                FN = FN + 1
        else:
            if y_pred[i]==1:
                FP = FP + 1
            else:
                TN = TN + 1

    return(TP, FP, TN, FN)



doc = get_data("Rules_binary.ods")

c=doc['Sheet1']

data=[x for x in c if x]


header=data[0]
data1=data[1:len(data)]


data2 =pd.DataFrame(data1, columns=header)

#data3=data2[['Sentences','Business function']]


data2['sentence']=data2['sentence'].str.lower()
data2['sentence']=data2['sentence'].str.strip()


#...........................splitting..................................

xw = data2['sentence'].tolist()
label=data2['lab'].tolist()
stop_words=set(stopwords.words("english"))
from nltk.stem.wordnet import WordNetLemmatizer
lmtzr = WordNetLemmatizer()
mm=[]
for i in xw:
    tokens=word_tokenize(i)
    tokens=[w.lower() for w in tokens]
    table=str.maketrans('','',string.punctuation)
    stripped=[w.translate(table) for w in tokens]
    words=[word for word in stripped if word.isalpha()]
#    filtered_sentence = [w for w in words if not w in stop_words]
    xx=[lmtzr.lemmatize(w) for w in words]
    mm.append(xx)


X_train,X_test,y_train,y_test=train_test_split(mm,label,test_size=0.2)



ones=sum(y_train)
zeros=len(y_train)-ones
class_weight = {0:(1/zeros),1: (1/ones)}



#.....................GENERATING word embeddings and its MATRIX...............................   


EMBEDDING_DIM=50

from gensim.models import FastText
model_f = FastText(X_train, size=EMBEDDING_DIM, window=5, 
                   min_count=3, workers=4,sg=1,negative=2)

words=list(model_f.wv.vocab)

filename='attention.txt'
model_f.wv.save_word2vec_format(filename,binary=False)
  
import numpy as np


import os
embeddings_index={}
f=open(os.path.join('','attention.txt'),encoding="utf-8")
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:])
    embeddings_index[word]=coefs
f.close()



from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

  
tokenizer_obj=Tokenizer()
tokenizer_obj.fit_on_texts(mm)
sequences=tokenizer_obj.texts_to_sequences(mm)
  
word_index=tokenizer_obj.word_index
    
num_words=len(word_index)+1    
embedding_matrix=np.zeros((num_words,EMBEDDING_DIM))
#max_length=max([len(s.split()) for s in mm])


for word,i in word_index.items():
    if i>num_words:
        continue
    embedding_vector=embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i]=embedding_vector
        
e=embedding_matrix[1:len(embedding_matrix)]
#m_pad=pad_sequences(sequences,maxlen=max_length)



#..................splitting training and validation set.......................

input_sentence_length=35
   
X_train_tokens=tokenizer_obj.texts_to_sequences(X_train)
X_test_tokens=tokenizer_obj.texts_to_sequences(X_test)

vocab_size=len(tokenizer_obj.word_index)+1

X_train_pad=pad_sequences(X_train_tokens,maxlen=input_sentence_length,
                          padding='post')

X_test_pad=pad_sequences(X_test_tokens,maxlen=input_sentence_length,
                         padding='post')
#
#X_train_pad=np.reshape(X_train_pad,(X_train_pad.shape[0], 1, X_train_pad.shape[1]))
#X_test_pad=np.reshape(X_test_pad,(X_test_pad.shape[0], 1, X_test_pad.shape[1]))
#

#..................building Sequntial model BiLSTM..........................


embedding_layer=Embedding(num_words,EMBEDDING_DIM,
                              embeddings_initializer=Constant(e),
                              input_length=input_sentence_length,
                              trainable=True)


word_input = Input(shape=(input_sentence_length,), dtype='float32')
word_sequences = embedding_layer(word_input)

#x = SpatialDropout1D(0.2)(word_sequences)
x = Bidirectional(LSTM(32,dropout=0.7,recurrent_dropout=0.7,
                       return_sequences=True))(word_sequences)

#x = Bidirectional(LSTM(32,dropout=0.7,recurrent_dropout=0.7,
#                      return_sequences=False))(x)

#y = Bidirectional(LSTM(32,dropout=0.3,recurrent_dropout=0.3, 
#                       return_sequences=True))(x)

#x = LSTM(64,dropout=0.5,recurrent_dropout=0.5,
#                       return_sequences=False)(word_sequences)

atten_1 = Attention(input_sentence_length)(x)  
# skip connect
#atten_2 = Attention(input_sentence_length)(y)

#avg_pool = GlobalAveragePooling1D()(y)
#conc = concatenate([atten_1, atten_2])#, avg_pool, max_pool])
#dd=Dense(32, activation='relu')(atten_1)

preds = Dense(1, activation='sigmoid')(atten_1)

model = Model(inputs=word_input, outputs=preds)

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['acc'])

#print(model.summary())

#--------------------------------------------------------------------------------------

h=model.fit(X_train_pad,y_train,batch_size=64,class_weight=class_weight,
           epochs=50,validation_data=(X_test_pad,y_test),verbose=1)

print(model.summary())

import matplotlib.pyplot as plt

plt.plot(h.history['acc'])
plt.plot(h.history['val_acc'])
plt.title('model BiLSTM=32')
plt.ylabel('accuracy')
plt.xlabel('Epochs')
plt.legend(['Training_set', 'Validation_set'], loc='upper left')
plt.show()


predictions = model.predict(X_test_pad)

pred1=pd.DataFrame(predictions)

pred2 = np.where(pred1 > 0.5, 1, 0)

pred3=list(pred2)
pred4=list(itertools.chain(*pred3))


from sklearn.metrics import confusion_matrix

cm=confusion_matrix(y_test, pred4)

TP, FP, TN, FN = confusion_matrix_values(y_test, pred4)
            
val_acc=(TP+TN)/(TP+FP+FN+TN+0.0001)
train_acc=(h.history['acc'][-1])*100

val_ac=(val_acc*100)
cm_acc=np.trace(cm)/np.sum(cm)

predicted_ones=sum(pred4)
ones_test_set=sum(y_test)

prec=TP/predicted_ones*100
recall=TP/ones_test_set*100


print('precision',prec)
print('recall   ',recall)
print('val_acc  ',val_ac)
print('train_acc',train_acc)

print(cm)
#print('cm_acc',cm_acc)



#.......................................................................


def Sorting(lst): 
    lst2 = sorted(lst, key=len) 
    return lst2

g_sorted = sorted(mm, key=len,reverse=True)

length=[]
for i in range(0,len(g_sorted)):
    x=len(g_sorted[i])
    length.append(x)


#for red lines

check1=int(5/100*len(length))

idx1=length[check1]



zx=[0,check1,check1,check1]
zy=[idx1,idx1,idx1,0]


#for coordinates
mx=[check1]
my=[idx1]

import matplotlib.pyplot as plt

plt.margins(0.005)
plt.plot(length)

plt.title('Tokenised sentences in decreasing order of their word lengths')
plt.ylabel('Sentence length')
plt.xlabel('sentence index ')
plt.legend(['5% pruned'],loc='upper right')
plt.plot(zx,zy,'r')
for i_mx, i_my in zip(mx, my):
    plt.text(i_mx, i_my, '({}, {})'.format(i_mx, i_my))
plt.show()


print('no. of sentences of lengths greater than',idx1, 'is =>',check1)



#.............................................................................










