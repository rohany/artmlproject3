'''
#Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.
'''

from __future__ import print_function
from keras.callbacks import LambdaCallback, ModelCheckpoint
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout, Bidirectional
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import io
import re
import math

# path = get_file(
#     'nietzsche.txt',
#     origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
# with io.open(path, encoding='utf-8') as f:
#     text = f.read().lower()

f = open('first_data.txt')
text = f.read().lower()
regexp = re.compile('[^\x09\x0A\x0D\x20-\x7f]')
text = regexp.sub('', text)
text = text.replace('\t', ' ')
# text = text.replace('\n', ' ')
text = re.sub(' +', ' ', text)
f.close()

text = text[:1500000]

print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

# cut the text in semi-redundant sequences of maxlen characters
# can increase this for more of a context?
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# def generate_batches(batch_size):
#     n_batches = int(math.ceil(len(sentences) / batch_size))
#     while True:
#         for i in range(n_batches):
# 	    x = np.zeros((batch_size, maxlen, len(chars)), dtype=np.bool)
# 	    y = np.zeros((batch_size, len(chars)), dtype=np.bool)
# 	    start = batch_size * i
# 	    for j in range(batch_size):
# 	    	sentence = sentences[start + j]
# 		for t, char in enumerate(sentence):
# 		    x[j, t, char_indices[char]] = 1
# 		y[j, char_indices[next_chars[start + j]]] = 1
# 	    yield x, y


# build the model: a single LSTM
print('Build model...')
model = Sequential()
# model.add(LSTM(512, return_sequences=True, input_shape=(maxlen, len(chars))))
model.add(LSTM(128, input_shape=(maxlen, len(chars))))
# model.add(Dropout(0.2))
# model.add(LSTM(512, return_sequences=False))
# model.add(Dropout(0.2))
model.add(Dense(len(chars), activation='softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(text) - maxlen - 1)
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)
	sys.stdout.write("\n")

        for i in range(400):
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

print_callback = LambdaCallback(on_epoch_end=on_epoch_end)
file_path = "weights-{epoch:02d}-{loss:.3f}.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor="loss", verbose=1, save_best_only=True, mode="min")

batch_size = 128

model.fit(x, y,
# model.fit_generator(generator=generate_batches(batch_size),
          epochs=60,
#	  steps_per_epoch=len(sentences) // batch_size // 100,
#	  steps_per_epoch=100000,
          callbacks=[print_callback, checkpoint])
