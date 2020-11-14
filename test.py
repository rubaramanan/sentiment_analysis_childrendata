import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import text_to_word_sequence, one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = load_model('model_(0.4490).h5')

sentence = 'I am an avid reader and picked this book up after my mom had gotten it from a friend'
word_sequences = set(text_to_word_sequence(sentence))
vector = one_hot(sentence, round(len(word_sequences) * 1.3))
word_arr = np.array(vector).astype('f')
word_arr = np.expand_dims(word_arr, axis=0)
test_data = pad_sequences(word_arr, maxlen=50, padding='post', truncating='post')

out = model.predict([test_data])
out = np.argmax(out)
print(out)

