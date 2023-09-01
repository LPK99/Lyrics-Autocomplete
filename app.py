import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import string, os
import tensorflow as tf

# keras module for building LSTM
from keras.utils import pad_sequences
from tensorflow.keras.layers import Embedding, Dropout, LSTM, Dense, Bidirectional
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
@st.cache_resource
def loadmodel():
    model = load_model('song_lyrics_generator_Not_One.h5')
    return model

def complete_this_song(input_text, next_words, model):
    df = pd.read_csv('lyrics-data.csv')
    df = df[df['language'] =='en']
    df = df[df['ALink'] == '/taylor-swift/']
    df.drop(['ALink','SName','SLink'],axis=1,inplace=True)
    # Tokenization
    tokenizer = Tokenizer()
    
    tokenizer.fit_on_texts(df['Lyric'].astype(str).str.lower())

    total_words = len(tokenizer.word_index)+1

    tokenized_sentences = tokenizer.texts_to_sequences(df['Lyric'].astype(str))
    input_sequences = list()
    for i in tokenized_sentences:
        for t in range(1, len(i)):
            n_gram_sequence = i[:t+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])


    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X, labels = input_sequences[:,:-1],input_sequences[:,-1]


    y = tf.keras.utils.to_categorical(labels, num_classes=total_words) # One hot encoding

    model = Sequential()

    model.add(Embedding(total_words, 40, input_length=max_sequence_len-1))
    model.add(Bidirectional(LSTM(250))) # 250 is the average number of words in a song
    model.add(Dropout(0.1)) # To overcome overfitting
    model.add(Dense(total_words, activation='softmax'))
    
    for _ in range(next_words):
        # for _ in... this is like a place holder, which upholds the syntax.
        # We use it when we don't want to use the variable, so we leave it empty.

        # Doing the same things to the input as we did when training the model
        token_list = tokenizer.texts_to_sequences([input_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_sequence_len-1, padding='pre')
        #predicted = model.predict_classes(token_list, verbose=0)
        predicted = np.argmax(model.predict(token_list), axis=-1)

        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                # Gets the word corresponding the the value predicted
                # [Converting from numeric to string again]
                output_word = word
                break
        input_text += " " + output_word
    st.write(input_text)


def main():
    st.title('Lyrics Generator')
    input = st.text_input("Enter your Lyrics")
    if st.button("Generate your song"):
        model = loadmodel()
        complete_this_song(input_text=input, next_words=80, model=model)
        
    

if __name__ == "__main__":
    main()
    