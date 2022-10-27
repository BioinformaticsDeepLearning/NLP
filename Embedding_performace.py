#Import important and necessary libraries#
from tqdm._tqdm_notebook import tqdm_notebook
import nltk
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, LSTM, Embedding,Bidirectional
import tensorflow
from tensorflow.compat.v1.keras.layers import CuDNNLSTM,CuDNNGRU
from tensorflow.keras.layers import Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn import metrics

#Removing unecessary terminology from Adverse reactions#
tqdm_notebook.pandas()
def text_preprocessing(df,col_name):
    column = col_name
    df[column] = df[column].progress_apply(lambda x:str(x).lower())
    df[column] = df[column].progress_apply(lambda x: th.cont_exp(x)) #you're -> you are; i'm -> i am
    df[column] = df[column].progress_apply(lambda x: th.remove_stopwords(x))
    df[column] = df[column].progress_apply(lambda x:th.spelling_correction(x))
    df[column] = df[column].progress_apply(lambda x: th.remove_special_chars(x))
    df[column] = df[column].progress_apply(lambda x: th.remove_accented_chars(x))
    df[column] = df[column].progress_apply(lambda x: th.make_base(x)) 
    return(df)

df = pd.read_csv("Vig.csv")
_df_ = text_preprocessing(df,df["Indication", "Drug" ,"Adversereaction"])

#NLTK for Calculating the word frequency#
words_list = []
for sentence in cleaned_df.Message:
    words_list.extend(nltk.word_tokenize(sentence))
freq_dist = nltk.FreqDist(words_list)
freq_dist.most_common(20)


X_train,X_test, y_train,y_test = train_test_split(_df_, test_size = 0.2,
                                                 ,random_state = 42)
                                                 
num_words = 20000 
tokenizer=Tokenizer(num_words,lower=True)
df_total = pd.concat([X_train, X_test], axis = 0)
tokenizer.fit_on_texts(df_total)


X_train_ =tokenizer.texts_to_sequences(X_train)
X_train_pad=pad_sequences(X_train_,maxlen=171,padding='post')
X_test_ = tokenizer.texts_to_sequences(X_test)
X_test_pad = pad_sequences(X_test_, maxlen = 171, padding = 'post')
                              

# modeling
emb_dim = 50 
model_gensim = Sequential()
model_gensim.add(Embedding(input_dim = ,
                          output_dim = emb_dim, 
                          input_length= X_train_pad.shape[1], 
                          weights = [gensim_weight_matrix],trainable = False))
model_gensim.add(Dropout(0.2))
model_gensim.add(Bidirectional(CuDNNLSTM(100,return_sequences=True)))
model_gensim.add(Dropout(0.2))
model_gensim.add(Bidirectional(CuDNNLSTM(200,return_sequences=True)))
model_gensim.add(Dropout(0.2))
model_gensim.add(Bidirectional(CuDNNLSTM(100,return_sequences=False)))
model_gensim.add(Dense(1, activation = 'sigmoid'))
model_gensim.compile(loss = 'binary_crossentropy', optimizer = 'adam',metrics = 'accuracy')

model_gensim.summary()
Stop = EarlyStopping(monitor = 'val_loss', mode = 'min', verbose = 1, patience = 5)
Checkpot = ModelCheckpoint('./model_gensim.h5', monitor = 'val_accuracy', mode = 'max', verbose = 1, save_best_only = True)

history_gensim = model_gensim.fit(X_train_pad,y_train, epochs = 25, batch_size = 120, validation_data=(X_test_pad, y_test),verbose = 1, callbacks= [Stop, Checkpot]  )

model_gensim.evaluate(X_test_pad, y_test) 
model.evaluate(X_test_pad, y_test)
y_pred = np.where(model.predict(X_test_pad)>.5,1,0)
print(metrics.classification_report(y_pred, y_test))
y_pred_gensim = np.where(model_gensim.predict(X_test_pad)>0.5,1,0)
print(metrics.classification_report(y_pred_gensim, y_test))
