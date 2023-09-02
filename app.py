import streamlit as st
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np



# keras module for building LSTM
from keras.utils import pad_sequences
from keras.preprocessing.text import Tokenizer


import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt

artist = ['the-beatles', 'john-mayer', 'taylor-swift']

@st.cache_resource
def loadmodel(model_name):
    return load_model(model_name)
@st.cache_data
def load_csv():
    return pd.read_csv('dataset/lyrics-data.csv')

def complete_this_song(input_text, next_words, artist_name):
    df = load_csv()
    df = df[df['language'] =='en']
    df = df[df['ALink'] == f'/{artist_name}/']
    df.drop(['ALink','SName','SLink'],axis=1,inplace=True)
    # Tokenization
    tokenizer = Tokenizer()
    
    tokenizer.fit_on_texts(df['Lyric'].astype(str).str.lower())

    tokenized_sentences = tokenizer.texts_to_sequences(df['Lyric'].astype(str))
    input_sequences = list()
    for i in tokenized_sentences:
        for t in range(1, len(i)):
            n_gram_sequence = i[:t+1]
            input_sequences.append(n_gram_sequence)

    max_sequence_len = max([len(x) for x in input_sequences])


    model = loadmodel(f'model/{artist_name}.h5')
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
    st.subheader(input_text)


def main():
    st.title('Lyrics Autocomplete')
    input = st.text_input("Enter your Lyrics")
    artist_selection = st.selectbox("Select your artist", artist)
    if st.button("Generate your song"):
        complete_this_song(input_text=input, next_words=80, artist_name=artist_selection)
        

if __name__ == "__main__":
    main()
    