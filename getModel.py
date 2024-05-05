import os
seed_value= 0
os.environ['PYTHONHASHSEED']=str(seed_value)

import keras 
from keras.layers import Input, Embedding, GRU, LSTM, Dense, Lambda, LeakyReLU, Dense, Concatenate, Flatten
from keras.layers import TimeDistributed, Bidirectional
from keras.layers import Layer, Flatten, Lambda
from keras.models import Model
from keras import activations, initializers, constraints
from keras.regularizers import l1, l2, l1_l2

from keras_preprocessing.sequence import pad_sequences#from keras.preprocessing.sequence import pad_sequences
import cv2
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import MaxPooling1D
from keras.layers import Dropout

from keras.layers import Conv1D
from keras.layers import Flatten

#<--get the reproducible results> 


import random
random.seed(seed_value)
import numpy as np
np.random.seed(seed_value)
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)

# from keras import backend as K
# session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)

opt=tf.keras.optimizers.legacy.Adam(lr=0.001, beta_1=0.5) #choose learning rate from {0.001, 0.0001, 0.00001}, here we choose 0.001 for optimal result


MAX_COMS_LENGTH = 120
embedding_dim = 100

loss = 'binary_crossentropy'

METRICS = [tf.keras.metrics.BinaryAccuracy(name='accuracy'),tf.keras.metrics.Precision(name='precision'),tf.keras.metrics.Recall(name='recall')]


@tf.custom_gradient
def grad_reverse(x):
	y = tf.identity(x)
	def custom_grad(dy):
		return -dy
	return y, custom_grad





def build_NeuralNet(embedding_matrix, word_index):
	#<====News Neural Network====>
	
	#<--First we prepare embedding layers for tweets-->
	embedding_layer_comments = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length = MAX_COMS_LENGTH, trainable = True, mask_zero=True)
	

	# channel 1
	inputs1 = Input(shape=(MAX_COMS_LENGTH,), dtype='int32')#(None, 120)
	embedding1 = embedding_layer_comments(inputs1)#(None, 120, 100)
	
	conv1 = Conv1D(filters=32, kernel_size=1, activation='relu')(embedding1)
	drop1 = Dropout(0.5)(conv1)
	pool1 = MaxPooling1D(pool_size=2)(drop1)
	flat1 = Flatten()(pool1)




	# channel 2
	inputs2 = Input(shape=(MAX_COMS_LENGTH,), dtype='int32')#(None, 120)
	embedding2 = embedding_layer_comments(inputs2)#(None, 120, 100)
	conv2 = Conv1D(filters=32, kernel_size=2, activation='relu')(embedding2)
	drop2 = Dropout(0.5)(conv2)
	pool2 = MaxPooling1D(pool_size=2)(drop2)
	flat2 = Flatten()(pool2)



	# channel 3
	inputs3 = Input(shape=(MAX_COMS_LENGTH,), dtype='int32')#(None, 120)
	embedding3 = embedding_layer_comments(inputs3)
	conv3 = Conv1D(filters=32, kernel_size=3, activation='relu')(embedding3)
	drop3 = Dropout(0.5)(conv3)
	pool3 = MaxPooling1D(pool_size=2)(drop3)
	flat3 = Flatten()(pool3)
	
	# channel 4
	inputs4 = Input(shape=(MAX_COMS_LENGTH,), dtype='int32')#(None, 120)
	embedding4 = embedding_layer_comments(inputs4)
	conv4 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding4)
	drop4 = Dropout(0.5)(conv4)
	pool4 = MaxPooling1D(pool_size=2)(drop4)
	flat4 = Flatten()(pool4)
	# merge
	concat = Concatenate(axis=1)
	merged = concat([flat1, flat2, flat3,flat4])


	#<----Crisis Tweet Classification------------>
		
	hidden_embed = Dense(512,activation='relu')(merged)

	#hidden_embed = Dense(512,activation='relu')(hidden_embed)

	preds = Dense(1, activation='sigmoid', name="main_output")(hidden_embed)

	#---Event Detection-------

	outputs=Lambda(grad_reverse, name="Gradient_Reversal")(merged)

	hidden_event = Dense(512, activation='relu')(outputs)
	
	event_label = Dense(3, activation='softmax')(hidden_event)

	
	model = Model(inputs = [inputs1, inputs2, inputs3, inputs4], outputs = [preds, event_label])

	print(model.summary())

	loss = ['binary_crossentropy']+['categorical_crossentropy']
	
	loss_weights = [1.4]+[1.4]

	#model.compile(optimizer=opt,
	 #             loss=loss,
	#	     metrics=METRICS)



	model.compile(optimizer='adam',
	              loss=loss,
	              loss_weights=loss_weights,
		      metrics=['accuracy', 'accuracy'])

	


	return model	


