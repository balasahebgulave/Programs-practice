# TypeError: Cannot interpret feed_dict key as Tensor: Tensor Tensor("Placeholder:0", shape=(1000, 10), dtype=float32) is not an element of this graph.
# 'Cannot interpret feed_dict key as Tensor: ' + e.args[0])
# solution 
    # This worked for me
    # from keras import backend as K
    # and after predicting my data i inserted this part of code
    # K.clear_session()
    
   
   
####libraries
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding
from keras.optimizers import RMSprop
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.utils as ku
from keras.models import Sequential
from sklearn.preprocessing import LabelEncoder
import pickle
from sklearn.externals import joblib
from keras import backend as K

# from config import PICKLE_FILE_PATH



def RNN_model(dfx,dfy,botId,userID):

	gle = LabelEncoder()
	genre_labels = gle.fit_transform(dfy)
	genre_mappings = {index: label for index, label in enumerate(gle.classes_)}

	# you need to create new dictionary to save sequential data in new file .pkl etc
	data_dict = {}
	for i in genre_mappings:
		data_dict[i] = genre_mappings[i]
	

	# train data x

	X = dfx
	classes = len(X) + 1
	max_words = 1000
	max_len = 150
	tok = Tokenizer(num_words=max_words)
	tok.fit_on_texts(X)
	sequences = tok.texts_to_sequences(X)
	sequences_matrix = sequence.pad_sequences(sequences,maxlen=max_len)
	



	# sequences_matrix
	# y_train
	
	label = ku.to_categorical(genre_labels , num_classes = classes)



	# model 
	
	def RNN():
	    model = Sequential()
	    model.add(Embedding(max_words, 10, input_length=max_len))
	    
	    model.add(LSTM(100))
	    model.add(Dropout(0.3))
	    
	    # Add Output Layer
	    model.add(Dense(classes , activation='softmax'))

	    model.compile(loss='categorical_crossentropy', optimizer='adam')
	    
	    return model

	model = RNN()
	

	# model.summary()
	model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
	
	# adding no of epochs to get better accuracy
	history=model.fit(sequences_matrix,label,epochs=100,verbose=2)
	

	# test text
	x_test = ['hi']
	sequences1 = tok.texts_to_sequences(x_test)
	sequences_matrix1 = sequence.pad_sequences(sequences1,maxlen=max_len)
	

	# prediction 
	pred=model.predict(sequences_matrix1)
	a=np.argmax(pred)
	


	# print(genre_mappings[a])
	print(genre_mappings)
	

	filename = "final_model.pickle"
	joblib.dump(model, open(filename, 'wb'))
	joblib.dump(data_dict , open('myfile.pickle', 'wb') )
	K.clear_session()




	# optional way of prediction using csv 
	# df = pd.Series(genre_mappings)
	# df.to_csv('file.csv' , index = None , header = True)	
	# print(df[0])




features = ['how many employees do you have?', 'employees', 'firm employees', 'body of company', 'company strength', 'heart of company', 'what is your productivity?', 'want to about team size', 'whats company  Strength', 'qwerty', 'asdfg', 'zxcvb', 'hi', 'hello', 'hi', 'hello', "what's up", 'hey', 'hi there', 'hello there', 'hi', 'hello', 'hey', "what's up", "get's started", 'wassup', 'yo', 'hie', 'hi', 'hello', 'hii', 'heyy', 'hi rnt', 'hello rnt', 'hi bot', 'hello bot', 'hi rnt', 'hello rnt', 'hi rnt bot', 'hello rnt bot', 'whats up', 'how are you?', 'how you doing?', 'how is it going for you?', 'how r you?', 'how are you doing?', 'how r u doing?', 'where were you born?', 'where is your home?', 'where do you live?', 'where is the headquarter?', 'where are you located?', "what's your birth place?", "what's your native place?", "what's your origin?", 'where are you from?', 'what is your address?', 'where are you now?', 'who are your parents?', 'who made you?', 'who is your father?', 'who is your mother?', 'who is your family?', 'can you tell me about yourself?', 'who are your folks?', 'who is your creator?', 'do you like coffee?', 'do you prefer', 'do you love coffee?', 'What can you do?', 'What do you do?', "What's your use?", 'How can you help me?', 'Why are you here?', 'Why they made you?', "What's your purpose?", 'What is your job?', 'What is your duty?', 'ok', 'Why Rabbit and tortoise', 'What is your moto?', 'Why this name?', 'What is the reason behind this name?', 'Why did you choose this name?', "What's the story behind this name?", 'How can we meet?', 'How can I get in touch with you?', 'How can we engage?', "let's meet", "let's have a conversation", 'How can I contact you', 'Where I can see you?', "What's your address?", 'Where is your office?', "What's your email address?", "What's your contact number?", 'can we have face to face conversation', "Where's your headquarter?", 'Can you schedule a meeting?', 'company details', 'give me your address', 'company address', 'contact details', "what's your address", 'tell me your location ', 'tell me your address', 'wheres this company ', 'whats the address of company', 'branch in india', 'branches across india', 'what is your name?', "what's your name?", 'your name', 'name', 'tell me your name?', 'your name please', 'can you tell me your name?', 'do you have name', 'who are you?', 'who r you?', 'what is your good name?', 'tell me about urself?', 'tell me about yourself?', 'who r u ?', 'Marketing', 'Any current openings?', 'Any open positions?', 'Any vacancy?', 'Current openings', 'current opening', 'opening', 'Tell me about current openings', 'Tell me about your internship', 'Do you have any workshop', 'Training services', 'deep learning', 'machin learning', 'artificial intelligence', 'intern', 'fresher', 'DL', 'ML', 'What technologies do you work in', 'Do you hire interns', 'full time jobs for interns', 'AI', 'bye', 'tata', 'good bye', 'ok bye', 'bye bye', 'farewell', 'thank you', 'thanks', 'thank u', "that's it", 'done', "i don't want any help.", 'It was nice talking to you', 'bbye', 'b-bye', 'BYee', 'Byee', 'problem', "i don't like you", 'not', "doesn't", 'I dont want any help.', "n't", "i don't need help", ' i dont wanna talk to you', 'nothing', 'No Problem', 'Stupid bot', 'idiot bot', 'foolish bot', 'dummy bot', 'u r stupid bot', 'You Are moron', 'Jerk', 'mad', 'crazy', 'mad', 'crazy', 'talking to you was horrible', 'shit', 'oh shit', 'why are you smiling', 'y r u smiling?', 'y r u sad ?', 'why r u sad?', "why're you laughing ?", 'why are you happy ?', 'what are you doing ?', 'what were you doing?', 'what do you do?', 'what r u doing? ', 'how do you work?', 'what is your work.', 'AI', 'Artificial Intelligence ', 'ai', '.ai', 'Tell me about the artificial intelligence', 'whats Al ', 'what is Al', 'whats Artifical Intelligence', 'establishment ', 'comapany details ', 'history company ', 'when did the company established', 'When company started', 'company partner', 'incorporated with', 'company assoicated with ', 'spouncer of company', 'company tie up', "What's your age?", 'whats your age', 'what is your age?', 'How long you have been working?', 'when they created you', 'how long you been around', 'how much experience do you have', 'How old are you', 'how old is your company', 'How do you know', 'how do u know everything', 'What services do you offer?', 'What services do you provide?', 'Do you deal with sales?', 'What products do you have?', 'In what your company works?', 'sales', 'Zero to One Services', 'ADM', 'Proposal Services', 'tell me about services', 'you r not human', 'r u human', 'are you human', 'Are you a robot?', 'r u a robot', 'are you bot?', 'Are you a chatbot', 'are u boy', 'women or girl ', 'man or boy ', 'what are you man or woman', 'cool', "that's cool", 'wow', 'nice one', 'I am bored', 'i m bored', "i'm bored?", 'tired', 'i m not fine', 'i m sick ', 'you are boring', 'u r boring', 'you are annoying ', 'you are bad ', 'really', 'is that so', 'actually', 'oh', 'ohh', 'yup', 'yea', 'yeah', 'yes', 'ok', 'okay', 'k', 'K', 'Oo', 'serious', 'seriously', 'sure', 'I am', "I'm", 'Im', 'am', 'This is ', 'my name is ', 'thats my name', 'Oh sorry', 'sorry', 'really sorry', 'i am sorry', 'ok sorry', 'sry', 'sory', 'so sorry', 'very sorry', 'Do you have any employment?', 'Do you provide any employment?', 'Are you hiring?', 'Are you accepting applications?', 'Do you have any job?', 'Any room for developer?', 'Any post?', 'Can I apply?', 'position', 'employ', 'vacant', 'place', 'How can i know your interview process?', 'interview rounds', 'How can i upload my resume ?', 'send resume ', 'send cv ', 'Interview timings', 'May i know the interview rounds', 'interview process', 'intervierw schedule', 'can send email to hr', 'can i send e mail to company  ', 'can i send email to ceo ?', 'interview structure', 'Job regarding ', 'jobs regarding position', 'Salary package for', 'Employment', 'employment', 'ADM', 'adm', 'what is adm?', "what's adm?", 'whats adm ?', 'Proposal services', 'Tell me about proposal services', 'Zero to One Services', '0/1', '0\\1', 'tell me a joke', 'joke', 'jokes', 'random fun', 'tell me something funny', 'funny joke', 'crack a joke', 'one more', 'whose the founder of the company', 'founder of company', 'HR  Manager ', 'who is the CEO of company ', 'Manager of company', 'who establish the company', 'who found rnt ?', 'who created this company ', 'who is creator of rnt ?', 'ceo', 'owner', 'whose your father', 'who is your mother', 'who are your parents', 'sex', 'sexual', 'sexy', 'intercourse', 'What project you have done', 'company Project ', 'Projects working on ', 'Current projects', 'what is rnt?', 'Rnt?', 'Rabbit and Tortoise', 'Company name ', 'name of company', 'tell me about your company', 'what is Rabbit and tortoise?', "what's rnt?", 'whats rnt', "company's name", 'rnt', 'about company', 'company', 'whats new', 'new stuff', 'new newz', 'company news ', 'workshop detail', 'may i know the office timing ', 'office timings', 'tell me the office timing ', 'when is the office open', 'office starts', 'when the office close', 'office holiday', 'office off days ', 'office on days', 'what time office opens', 'what time office close', 'what  time and day i can come', 'when can i come to office ', 'can i visit you anytime ', 'can i meet you any time', 'what time should i come ', 'tell me the timings of company', 'When are you open?', 'Are you open today?', 'How late are you open on weekends?', 'When do you close?', 'What time do you open tomorrow morning?', 'Are you open now?', 'Business hours.', 'How early can I drop in?', 'Tell me your opening hours.', 'What are your hours?', 'How late can I come in?'] 
labels = ['company strength', 'company strength', 'company strength', 'company strength', 'company strength', 'company strength', 'company strength', 'company strength', 'company strength', 'Default Fallback Intent', 'Default Fallback Intent', 'Default Fallback Intent', 'Default Welcome Intent', 'Default Welcome Intent', 'Default Welcome Intent', 'Default Welcome Intent', 'Default Welcome Intent', 'Default Welcome Intent', 'Default Welcome Intent', 'Default Welcome Intent', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Greeting', 'Default BOT Feeling', 'Default BOT Feeling', 'Default BOT Feeling', 'Default BOT Feeling', 'Default BOT Feeling', 'Default BOT Feeling', 'Default BOT Native', 'Default BOT Native', 'Default BOT Native', 'Default BOT Native', 'Default BOT Native', 'Default BOT Native', 'Default BOT Native', 'Default BOT Native', 'Default BOT Native', 'Default BOT Native', 'Default BOT Native', 'Default BOT Founders', 'Default BOT Founders', 'Default BOT Founders', 'Default BOT Founders', 'Default BOT Founders', 'Default BOT Founders', 'Default BOT Founders', 'Default BOT Founders', 'Default BOT Preference', 'Default BOT Preference', 'Default BOT Preference', 'What can you do', 'What can you do', 'What can you do', 'What can you do', 'What can you do', 'What can you do', 'What can you do', 'What can you do', 'What can you do', 'Fallback for what can I do', 'Company Questions', 'Company Questions', 'Company Questions', 'Company Questions', 'Company Questions', 'Company Questions', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Company details', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Default BOT Name', 'Marketing', 'Current openings', 'Current openings', 'Current openings', 'Current openings', 'Current openings', 'Current openings', 'Current openings', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Internship', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Goodbye', 'Default BOT Negativity Reply', 'Default BOT Negativity Reply', 'Default BOT Negativity Reply', 'Default BOT Negativity Reply', 'Default BOT Negativity Reply', 'Default BOT Negativity Reply', 'Default BOT Negativity Reply', 'Default BOT Negativity Reply', 'Default BOT Negativity Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'Default BOT Badword Reply', 'smiling', 'smiling', 'smiling', 'smiling', 'smiling', 'smiling', 'Default BOT Work', 'Default BOT Work', 'Default BOT Work', 'Default BOT Work', 'Default BOT Work', 'Default BOT Work', 'AI', 'AI', 'AI', 'AI', 'AI', 'AI', 'AI', 'AI', 'company foundation', 'company foundation', 'company foundation', 'company foundation', 'company foundation', 'company foundation', 'company foundation', 'company foundation', 'company foundation', 'company foundation', 'Default BOT Age', 'Default BOT Age', 'Default BOT Age', 'Default BOT Age', 'Default BOT Age', 'Default BOT Age', 'Default BOT Age', 'Default BOT Age', 'Default BOT Age', 'How do you know', 'How do you know', 'Sales', 'Sales', 'Sales', 'Sales', 'Sales', 'Sales', 'Sales', 'Sales', 'Sales', 'Sales', 'notHuman', 'notHuman', 'notHuman', 'notHuman', 'notHuman', 'notHuman', 'notHuman', 'notHuman', 'notHuman', 'notHuman', 'notHuman', 'Default BOT Cool', 'Default BOT Cool', 'Default BOT Cool', 'Default BOT Cool', 'I am Bored', 'I am Bored', 'I am Bored', 'I am Bored', 'I am Bored', 'I am Bored', 'I am Bored', 'I am Bored', 'I am Bored', 'I am Bored', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'really', 'Nice to know you', 'Nice to know you', 'Nice to know you', 'Nice to know you', 'Nice to know you', 'Nice to know you', 'Nice to know you', "It's okay. That's all right!", "It's okay. That's all right!", "It's okay. That's all right!", "It's okay. That's all right!", "It's okay. That's all right!", "It's okay. That's all right!", "It's okay. That's all right!", "It's okay. That's all right!", "It's okay. That's all right!", 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'Employment', 'ADM', 'ADM', 'ADM', 'ADM', 'ADM', 'Proposal services', 'Proposal services', 'Zero to One Services', 'Zero to One Services', 'Zero to One Services', 'Default BOT Jokes', 'Default BOT Jokes', 'Default BOT Jokes', 'Default BOT Jokes', 'Default BOT Jokes', 'Default BOT Jokes', 'Default BOT Jokes', 'Default BOT Jokes', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'company founder', 'Default BOT Reply Badtalk', 'Default BOT Reply Badtalk', 'Default BOT Reply Badtalk', 'Default BOT Reply Badtalk', 'Company Project', 'Company Project', 'Company Project', 'Company Project', 'company name', 'company name', 'company name', 'company name', 'company name', 'company name', 'company name', 'company name', 'company name', 'company name', 'company name', 'company name', 'company name', 'whats new', 'whats new', 'whats new', 'whats new', 'whats new', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day', 'office time and day'] 


RNN_model(features , labels , 1 , 2)







# Run following code for prediction


import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing import sequence
from sklearn.externals import joblib
import tensorflow as tf
from keras import backend as K



def pred(message):
	max_words = 1000
	max_len = 150
	tok = Tokenizer(num_words=max_words)
	f = open('final_model.pickle' , 'rb')
	Model = pickle.load(f)
	f.close()
	x_test = message
	gle = LabelEncoder()


	# genre_mappings = {index: label for index, label in enumerate(gle.classes_)}
	sequences1 = tok.texts_to_sequences(x_test)
	sequences_matrix1 = sequence.pad_sequences(sequences1,maxlen=max_len)
	


	# graph = tf.get_default_graph()              
	# with graph.as_default():                    
	# 	predictions = Model.predict(sequences_matrix1)



	predictions = Model.predict(sequences_matrix1)
	b=np.argmax(predictions)
	
	file = open('myfile.pickle','rb')
	data = pickle.load(file)
	print(data[b])



	# optional  way of predicting RNN result
	# df = pd.read_csv('file.csv')
	# print(df[b])



pred(['hi'])
