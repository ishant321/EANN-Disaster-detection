import os
os.environ['TF_ENABLE_ONEDNN_OPTS']='0'
os.environ['TF_USE_LEGACY_KERAS']='True'

import pandas as pd
import json
import re	
import string
from ast import literal_eval as le
from keras_preprocessing.sequence import pad_sequences#from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
import getModel as gM

MAX_COMS_LENGTH = 120
TEST_SPLIT = 0.25	
embedding_dim = 100
batch_size = 16#8#32
Epochs = 10#180#25#60#105#95#270

def fit_on_news_and_comments(train_x, test_x):
	
	comments = []
	comments.extend(train_x)
	comments.extend(test_x)
	tokenizer = Tokenizer(num_words=20000)
	
	
	tokenizer.fit_on_texts(comments)
	VOCABULARY_SIZE = len(tokenizer.word_index) + 1
	#print(" len of all_text:", len(all_text))	
	print("vocab_size: ", VOCABULARY_SIZE)
	#input("wait..")
	
	reverse_word_index = {value: key for key, value in tokenizer.word_index.items()}

	return VOCABULARY_SIZE, tokenizer, reverse_word_index



def performCleaning(tweet):
	#https://www.kaggle.com/code/redwankarimsony/nlp-101-tweet-sentiment-analysis-preprocessing/notebook
	tweet = re.sub(r'\bRT\b', '', tweet) #remove RT

	tweet = re.sub(r'#', '', tweet) #remove hashtags
	tweet = re.sub(r'@', '', tweet) #remove mentions
	
	#tweet =  re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
	tweet = re.sub(r'http\S+', '', tweet) #remove hyperlinks
	tweet = re.sub(r'[0-9]', '', tweet) # remove numbers
	
	tweet = re.sub("'", "", tweet)
	
	
	STOP_WORDS = ['the', 'a', 'an']	
	tweet = ' '.join([word for word in tweet.split() if word not in STOP_WORDS and word not in string.punctuation])

	


	'''	
	#<==remove unwanted ... from text ===>	
	x = tweet.split('.')
	y = [ele.strip() for ele in x if ele]
	tweet = ' '.join(y)
	#<===================================>
	tweet = tweet.replace(',','') #remove comma
	
	
	#<====Lemmatization===>
	tweet_tokens = tokenizer.tokenize(tweet)
	lemmaL = []
	for word in tweet_tokens:
		lemmaL.append(wordnet_lemmatizer.lemmatize(word))
	tweet = ' '.join(lemmaL)
	'''	
	return tweet



def build_Embedding_Matrix(glove, t, aff_dim=80):
	
	embeddings_index = {}
	f = open(glove, encoding="utf-8")   
	for line in f:
		#print(line)
		#input(".......wait....")
		values = line.split()
		word = values[0]
		coefs = np.asarray(values[1:], dtype='float32')
		embeddings_index[word] = coefs
	f.close()

	print('Loaded %s word vectors.' % len(embeddings_index))
	word_index = t.word_index
	embedding_matrix = np.random.random((len(word_index) + 1, embedding_dim))
	for word, i in word_index.items():
		embedding_vector = embeddings_index.get(word) # embeddings_index[word]
		if embedding_vector is not None:
			embedding_matrix[i] = embedding_vector
	#return embedding_matrix

	print("shape of embedding_matrix: ", embedding_matrix.shape)
	
	return embedding_matrix, word_index


def _encode_comments(comments, t):

	#encoded_texts = np.zeros((len(comments), MAX_COMS_COUNT, MAX_COMS_LENGTH), dtype='int32')
	#for i, text in enumerate(comments):
		#encoded_text = np.array(pad_sequences(t.texts_to_sequences(comments),maxlen=MAX_COMS_LENGTH, padding='post', truncating='post', value=0))[:MAX_COMS_COUNT]
		#encoded_texts[i][:len(encoded_text)] = encoded_text
	embedded_sentences = t.texts_to_sequences(comments)
	padded_sentences = pad_sequences(embedded_sentences, maxlen=MAX_COMS_LENGTH, padding='post',truncating='post', value=0)

	#return encoded_texts
	return padded_sentences



def dataprepare(filename, event_name):
	if filename == "earthquake.csv":
		di_label = {"Off-topic":0, "on-topic":1}
	else:	
		di_label = {"off-topic":0, "on-topic":1}
	df = pd.read_csv(filename, delimiter = ",")
	
	df['tweet_clean'] =  df['tweet'].apply(lambda x:performCleaning(x))
	df['label_encoded'] =  df['label'].apply(lambda x:di_label[x])
	df['event_name'] = [event_name for i in range(df.shape[0])]
	return df

if __name__ == "__main__":
	di_event_label = {"bombing":0, "earthquake":1, "floods":2}
	di_encoded_event_label = {0:[0,0,1],1:[0,1,0],2:[1,0,0]}
	df_bomb = dataprepare("bombing.csv","bombing")
	df_earth = dataprepare("earthquake.csv","earthquake")
	df_flood = dataprepare("floods.csv","floods")
	
	df = pd.concat([df_bomb, df_earth, df_flood])
	print(df.head())
	input("df")


	df['event_label'] =  df['event_name'].apply(lambda x:di_event_label[x])

	df['event_label'] =  df['event_label'].apply(lambda x:di_encoded_event_label[x])

	print(list(set(df['event_name'].tolist())))
	#print(df['event_label'].tolist())

	print(df.shape)
	print(df.head())
	print(df.columns)

	#input("wait....")
	
	contents = df["tweet_clean"].tolist()
	labels = np.asarray(df["label_encoded"].tolist())
	labelsE = np.asarray(df["event_label"].tolist())
	
	train_x, test_x, train_y, test_y, train_e, test_e = train_test_split(contents, labels, labelsE, test_size=TEST_SPLIT, random_state=42,stratify=labels)

	print(train_y.shape)
	print(test_y.shape)
	
	
	print("===Start Work on Embedding===")
	VOCABULARY_SIZE, t, reverse_word_index = fit_on_news_and_comments(train_x, test_x)
	embedding_matrix, word_index = build_Embedding_Matrix("glove.6B.100d.txt", t, aff_dim=80)
	model = gM.build_NeuralNet(embedding_matrix, word_index)
	
	encoded_train_c = _encode_comments(train_x,t)
	encoded_test_c = _encode_comments(test_x,t)
	print("shape of encoded_train_c: ",encoded_train_c.shape)
	print("shape of encoded_test_c: ",encoded_test_c.shape)
	
	#input("---------------------------wait-------------------------------------")
	history = model.fit([encoded_train_c,encoded_train_c,encoded_train_c,encoded_train_c], y= [train_y, train_e], validation_split=0.2, batch_size=batch_size, epochs=Epochs, verbose=2)#,callbacks=[callback])

	#input("training is done...")

	y_predicted = model.predict([encoded_test_c, encoded_test_c, encoded_test_c,encoded_test_c])
    	
    
    
    
    
   

	print(type(y_predicted))




	y_predicted=y_predicted[0]
	y_predicted = y_predicted.flatten()

	y_predicted = np.where(y_predicted > 0.5, 1, 0)


	#print(f"type(y_predicted): {type(y_predicted)}")
	#print(f"type(test_y): {type(test_y)}")
	
	print(y_predicted)
	print("=============================================")
	print(test_y)
	

	from sklearn.metrics import confusion_matrix, classification_report
	cm = confusion_matrix(test_y, y_predicted)

	print(cm)

	print(classification_report(test_y, y_predicted))