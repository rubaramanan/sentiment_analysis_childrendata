import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Softmax, Embedding, SpatialDropout1D
from tensorflow.keras.utils import to_categorical

df = pd.read_excel('FYPdatasets.xlsx', sheet_name='Sheet1')
print(len(df))


def labeling(age_grp, sent):
    # encoding
    if age_grp == 'above 5 year' and sent == 'positive':
        a = 0
    elif age_grp == 'above 5 year' and sent == 'negative':
        a = 1
    elif age_grp == 'above 5 year' and sent == 'neutral':
        a = 2
    elif age_grp == 'above 8 year' and sent == 'positive':
        a = 3
    elif age_grp == 'above 8 year' and sent == 'negative':
        a = 4
    elif age_grp == 'above 8 year' and sent == 'neutral':
        a = 5
    elif age_grp == 'above 10 year' and sent == 'positive':
        a = 6
    elif age_grp == 'above 10 year' and sent == 'negative':
        a = 7
    elif age_grp == 'above 10 year' and sent == 'neutral':
        a = 8
    else:
        a = 9
    return a


df['label'] = df[['Age_group', 'Sentiment']].apply(lambda a: labeling(a.Age_group, a.Sentiment), axis=1).astype('f')


# print(df)


def preprocessing(sentence):
    # divide sentences as each words using(text to word sequences)
    words = set(text_to_word_sequence(sentence))  # remove duplicate words
    vocab_size = len(words)
    # words to numeric value(vector)
    results = one_hot(sentence, round(vocab_size * 1.3))
    return results


df['words'] = df['Sentence'].apply(preprocessing)
max_len = max(df['words'].apply(len)) + 10
word_array = df.iloc[:, 4].values

x_train = pad_sequences(word_array, 50, dtype='f', padding='post', truncating='post')
y_train = to_categorical(df['label'].values, num_classes=10)

model = Sequential()
model.add(Embedding(50, 128))
model.add(LSTM(256, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(256))
model.add(Dense(10, activation='sigmoid'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())

model.fit(x_train, y_train, epochs=50)
model.save('model_(0.4490).h5')

