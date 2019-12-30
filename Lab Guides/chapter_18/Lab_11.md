Chapter 18
How to Develop a Character-Based
Neural Language Model
A language model predicts the next word in the sequence based on the specific words that have
come before it in the sequence. It is also possible to develop language models at the character
level using neural networks. The benefit of character-based language models is their small
vocabulary and flexibility in handling any words, punctuation, and other document structure.
This comes at the cost of requiring larger models that are slower to train. Nevertheless, in the
field of neural language models, character-based models offer a lot of promise for a general,
flexible and powerful approach to language modeling. In this tutorial, you will discover how to
develop a character-based neural language model. After completing this tutorial, you will know:
 How to prepare text for character-based language modeling.
 How to develop a character-based language model using LSTMs.
 How to use a trained character-based language model to generate text.

Let’s get started.

18.1

Tutorial Overview

This tutorial is divided into the following parts:
1. Sing a Song of Sixpence
2. Data Preparation
3. Train Language Model
4. Generate Text

197

18.2. Sing a Song of Sixpence

18.2

198

Sing a Song of Sixpence

The nursery rhyme Sing a Song of Sixpence is well known in the west. The first verse is common,
but there is also a 4 verse version that we will use to develop our character-based language
model. It is short, so fitting the model will be fast, but not so short that we won’t see anything
interesting. The complete 4 verse version we will use as source text is listed below.
Sing a song of sixpence,
A pocket full of rye.
Four and twenty blackbirds,
Baked in a pie.
When the pie was opened
The birds began to sing;
Wasn't that a dainty dish,
To set before the king.
The king was in his counting house,
Counting out his money;
The queen was in the parlour,
Eating bread and honey.
The maid was in the garden,
Hanging out the clothes,
When down came a blackbird
And pecked off her nose.

Listing 18.1: Sing a Song of Sixpence nursery rhyme.
Copy the text and save it in a new file in your current working directory with the file name
rhyme.txt.

18.3

Data Preparation

The first step is to prepare the text data. We will start by defining the type of language model.

18.3.1

Language Model Design

A language model must be trained on the text, and in the case of a character-based language
model, the input and output sequences must be characters. The number of characters used
as input will also define the number of characters that will need to be provided to the model
in order to elicit the first predicted character. After the first character has been generated, it
can be appended to the input sequence and used as input for the model to generate the next
character.
Longer sequences offer more context for the model to learn what character to output next
but take longer to train and impose more burden on seeding the model when generating text.
We will use an arbitrary length of 10 characters for this model. There is not a lot of text, and
10 characters is a few words. We can now transform the raw text into a form that our model
can learn; specifically, input and output sequences of characters.

18.3. Data Preparation

18.3.2

199

Load Text

We must load the text into memory so that we can work with it. Below is a function named
load doc() that will load a text file given a filename and return the loaded text.
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text

Listing 18.2: Function to load a document into memory.
We can call this function with the filename of the nursery rhyme rhyme.txt to load the text
into memory. The contents of the file are then printed to screen as a sanity check.
# load text
raw_text = load_doc('rhyme.txt')
print(raw_text)

Listing 18.3: Load the document into memory.

18.3.3

Clean Text

Next, we need to clean the loaded text. We will not do much to it on this example. Specifically,
we will strip all of the new line characters so that we have one long sequence of characters
separated only by white space.
# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)

Listing 18.4: Tokenize the loaded document.
You may want to explore other methods for data cleaning, such as normalizing the case to
lowercase or removing punctuation in an effort to reduce the final vocabulary size and develop a
smaller and leaner model.

18.3.4

Create Sequences

Now that we have a long list of characters, we can create our input-output sequences used to
train the model. Each input sequence will be 10 characters with one output character, making
each sequence 11 characters long. We can create the sequences by enumerating the characters
in the text, starting at the 11th character at index 10.
# organize into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
# select sequence of tokens
seq = raw_text[i-length:i+1]

18.3. Data Preparation

200

# store
sequences.append(seq)
print('Total Sequences: %d' % len(sequences))

Listing 18.5: Convert text into fixed-length sequences.
Running this snippet, we can see that we end up with just under 400 sequences of characters
for training our language model.
Total Sequences: 399

Listing 18.6: Example output of converting text into fixed-length sequences.

18.3.5

Save Sequences

Finally, we can save the prepared data to file so that we can load it later when we develop our
model. Below is a function save doc() that, given a list of strings and a filename, will save the
strings to file, one per line.
# save tokens to file, one dialog per line
def save_doc(lines, filename):
data = '\n'.join(lines)
file = open(filename, 'w')
file.write(data)
file.close()

Listing 18.7: Function to save sequences to file.
We can call this function and save our prepared sequences to the filename char sequences.txt
in our current working directory.
# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)

Listing 18.8: Call function to save sequences to file.

18.3.6

Complete Example

Tying all of this together, the complete code listing is provided below.
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# save tokens to file, one dialog per line
def save_doc(lines, filename):
data = '\n'.join(lines)
file = open(filename, 'w')

18.4. Train Language Model

201

file.write(data)
file.close()
# load text
raw_text = load_doc('rhyme.txt')
print(raw_text)
# clean
tokens = raw_text.split()
raw_text = ' '.join(tokens)
# organize into sequences of characters
length = 10
sequences = list()
for i in range(length, len(raw_text)):
# select sequence of tokens
seq = raw_text[i-length:i+1]
# store
sequences.append(seq)
print('Total Sequences: %d' % len(sequences))
# save sequences to file
out_filename = 'char_sequences.txt'
save_doc(sequences, out_filename)

Listing 18.9: Complete example of preparing the text data.
Run the example to create the char sequences.txt file. Take a look inside you should see
something like the following:
Sing a song
ing a song
ng a song o
g a song of
a song of
a song of s
song of si
song of six
ong of sixp
ng of sixpe
...

Listing 18.10: Sample of the output file.
We are now ready to train our character-based neural language model.

18.4

Train Language Model

In this section, we will develop a neural language model for the prepared sequence data. The
model will read encoded characters and predict the next character in the sequence. A Long
Short-Term Memory recurrent neural network hidden layer will be used to learn the context
from the input sequence in order to make the predictions.

18.4.1

Load Data

The first step is to load the prepared character sequence data from char sequences.txt. We
can use the same load doc() function developed in the previous section. Once loaded, we split

18.4. Train Language Model

202

the text by new line to give a list of sequences ready to be encoded.
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')

Listing 18.11: Load the prepared text data.

18.4.2

Encode Sequences

The sequences of characters must be encoded as integers. This means that each unique character
will be assigned a specific integer value and each sequence of characters will be encoded as a
sequence of integers. We can create the mapping given a sorted set of unique characters in the
raw input data. The mapping is a dictionary of character values to integer values.
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))

Listing 18.12: Create a mapping between chars and integers.
Next, we can process each sequence of characters one at a time and use the dictionary
mapping to look up the integer value for each character.
sequences = list()
for line in lines:
# integer encode line
encoded_seq = [mapping[char] for char in line]
# store
sequences.append(encoded_seq)

Listing 18.13: Integer encode sequences of characters.
The result is a list of integer lists. We need to know the size of the vocabulary later. We can
retrieve this as the size of the dictionary mapping.
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)

Listing 18.14: Summarize the size of the vocabulary.
Running this piece, we can see that there are 38 unique characters in the input sequence
data.
Vocabulary Size: 38

Listing 18.15: Example output from summarizing the size of the vocabulary.

18.4. Train Language Model

18.4.3

203

Split Inputs and Output

Now that the sequences have been integer encoded, we can separate the columns into input and
output sequences of characters. We can do this using a simple array slice.
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]

Listing 18.16: Split sequences into input and output elements.
Next, we need to one hot encode each character. That is, each character becomes a vector as
long as the vocabulary (38 elements) with a 1 marked for the specific character. This provides
a more precise input representation for the network. It also provides a clear objective for the
network to predict, where a probability distribution over characters can be output by the model
and compared to the ideal case of all 0 values with a 1 for the actual next character. We can
use the to categorical() function in the Keras API to one hot encode the input and output
sequences.
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)
y = to_categorical(y, num_classes=vocab_size)

Listing 18.17: Convert sequences into a format ready for training.
We are now ready to fit the model.

18.4.4

Fit Model

The model is defined with an input layer that takes sequences that have 10 time steps and 38
features for the one hot encoded input sequences. Rather than specify these numbers, we use
the second and third dimensions on the X input data. This is so that if we change the length of
the sequences or size of the vocabulary, we do not need to change the model definition. The
model has a single LSTM hidden layer with 75 memory cells, chosen with a little trial and
error. The model has a fully connected output layer that outputs one vector with a probability
distribution across all characters in the vocabulary. A softmax activation function is used on
the output layer to ensure the output has the properties of a probability distribution.
# define the model
def define_model(X):
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize defined model
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)
return model

Listing 18.18: Define the language model.
The model is learning a multiclass classification problem, therefore we use the categorical log
loss intended for this type of problem. The efficient Adam implementation of gradient descent
is used to optimize the model and accuracy is reported at the end of each batch update. The

18.4. Train Language Model

204

model is fit for 100 training epochs, again found with a little trial and error. Running this prints
a summary of the defined network as a sanity check.
_________________________________________________________________
Layer (type)
Output Shape
Param #
=================================================================
lstm_1 (LSTM)
(None, 75)
34200
_________________________________________________________________
dense_1 (Dense)
(None, 38)
2888
=================================================================
Total params: 37,088
Trainable params: 37,088
Non-trainable params: 0
_________________________________________________________________

Listing 18.19: Example output from summarizing the defined model.
A plot the defined model is then saved to file with the name model.png.

Figure 18.1: Plot of the defined character-based language model.

18.4.5

Save Model

After the model is fit, we save it to file for later use. The Keras model API provides the save()
function that we can use to save the model to a single file, including weights and topology
information.
# save the model to file
model.save('model.h5')

Listing 18.20: Save the fit model to file.
We also save the mapping from characters to integers that we will need to encode any input
when using the model and decode any output from the model.
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))

Listing 18.21: Save the mapping of chars to integers to file.

18.4. Train Language Model

18.4.6

205

Complete Example

Tying all of this together, the complete code listing for fitting the character-based neural
language model is listed below.
from
from
from
from
from
from
from

numpy import array
pickle import dump
keras.utils import to_categorical
keras.utils.vis_utils import plot_model
keras.models import Sequential
keras.layers import Dense
keras.layers import LSTM

# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# define the model
def define_model(X):
model = Sequential()
model.add(LSTM(75, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(vocab_size, activation='softmax'))
# compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize defined model
model.summary()
plot_model(model, to_file='model.png', show_shapes=True)
return model
# load
in_filename = 'char_sequences.txt'
raw_text = load_doc(in_filename)
lines = raw_text.split('\n')
# integer encode sequences of characters
chars = sorted(list(set(raw_text)))
mapping = dict((c, i) for i, c in enumerate(chars))
sequences = list()
for line in lines:
# integer encode line
encoded_seq = [mapping[char] for char in line]
# store
sequences.append(encoded_seq)
# vocabulary size
vocab_size = len(mapping)
print('Vocabulary Size: %d' % vocab_size)
# separate into input and output
sequences = array(sequences)
X, y = sequences[:,:-1], sequences[:,-1]
sequences = [to_categorical(x, num_classes=vocab_size) for x in X]
X = array(sequences)

18.5. Generate Text

206

y = to_categorical(y, num_classes=vocab_size)
# define model
model = define_model(X)
# fit model
model.fit(X, y, epochs=100, verbose=2)
# save the model to file
model.save('model.h5')
# save the mapping
dump(mapping, open('mapping.pkl', 'wb'))

Listing 18.22: Complete example of training the language model.
Running the example might take one minute. You will see that the model learns the problem
well, perhaps too well for generating surprising sequences of characters.
...
Epoch 96/100
0s - loss: 0.2193
Epoch 97/100
0s - loss: 0.2124
Epoch 98/100
0s - loss: 0.2054
Epoch 99/100
0s - loss: 0.1982
Epoch 100/100
0s - loss: 0.1910

- acc: 0.9950
- acc: 0.9950
- acc: 0.9950
- acc: 0.9950
- acc: 0.9950

Listing 18.23: Example output from training the language model.
At the end of the run, you will have two files saved to the current working directory,
specifically model.h5 and mapping.pkl. Next, we can look at using the learned model.

18.5

Generate Text

We will use the learned language model to generate new sequences of text that have the same
statistical properties.

18.5.1

Load Model

The first step is to load the model saved to the file model.h5. We can use the load model()
function from the Keras API.
# load the model
model = load_model('model.h5')

Listing 18.24: Load the saved model.
We also need to load the pickled dictionary for mapping characters to integers from the file
mapping.pkl. We will use the Pickle API to load the object.
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))

Listing 18.25: Load the saved mapping from chars to integers.
We are now ready to use the loaded model.

18.5. Generate Text

18.5.2

207

Generate Characters

We must provide sequences of 10 characters as input to the model in order to start the generation
process. We will pick these manually. A given input sequence will need to be prepared in the
same way as preparing the training data for the model. First, the sequence of characters must
be integer encoded using the loaded mapping.
# encode the characters as integers
encoded = [mapping[char] for char in in_text]

Listing 18.26: Encode input text to integers.
Next, the integers need to be one hot encoded using the to categorical() Keras function.
# one hot encode
encoded = to_categorical(encoded, num_classes=len(mapping))

Listing 18.27: One hot encode the integer encoded text.
We can then use the model to predict the next character in the sequence. We use
predict classes() instead of predict() to directly select the integer for the character with
the highest probability instead of getting the full probability distribution across the entire set of
characters.
# predict character
yhat = model.predict_classes(encoded, verbose=0)

Listing 18.28: Predict the next character in the sequence.
We can then decode this integer by looking up the mapping to see the character to which it
maps.
out_char = ''
for char, index in mapping.items():
if index == yhat:
out_char = char
break

Listing 18.29: Map the predicted integer back to a character.
This character can then be added to the input sequence. We then need to make sure that the
input sequence is 10 characters by truncating the first character from the input sequence text.
We can use the pad sequences() function from the Keras API that can perform this truncation
operation. Putting all of this together, we can define a new function named generate seq()
for using the loaded model to generate new sequences of text.
# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
in_text = seed_text
# generate a fixed number of characters
for _ in range(n_chars):
# encode the characters as integers
encoded = [mapping[char] for char in in_text]
# truncate sequences to a fixed length
encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
# one hot encode
encoded = to_categorical(encoded, num_classes=len(mapping))

18.5. Generate Text

208

# predict character
yhat = model.predict_classes(encoded, verbose=0)
# reverse map integer to character
out_char = ''
for char, index in mapping.items():
if index == yhat:
out_char = char
break
# append to input
in_text += char
return in_text

Listing 18.30: Function to predict a sequence of characters given seed text.

18.5.3

Complete Example

Tying all of this together, the complete example for generating text using the fit neural language
model is listed below.
from
from
from
from
from

pickle import load
numpy import array
keras.models import load_model
keras.utils import to_categorical
keras.preprocessing.sequence import pad_sequences

# generate a sequence of characters with a language model
def generate_seq(model, mapping, seq_length, seed_text, n_chars):
in_text = seed_text
# generate a fixed number of characters
for _ in range(n_chars):
# encode the characters as integers
encoded = [mapping[char] for char in in_text]
# truncate sequences to a fixed length
encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
# one hot encode
encoded = to_categorical(encoded, num_classes=len(mapping))
# predict character
yhat = model.predict_classes(encoded, verbose=0)
# reverse map integer to character
out_char = ''
for char, index in mapping.items():
if index == yhat:
out_char = char
break
# append to input
in_text += out_char
return in_text
# load the model
model = load_model('model.h5')
# load the mapping
mapping = load(open('mapping.pkl', 'rb'))
# test start of rhyme
print(generate_seq(model, mapping, 10, 'Sing a son', 20))
# test mid-line

18.6. Further Reading

209

print(generate_seq(model, mapping, 10, 'king was i', 20))
# test not in original
print(generate_seq(model, mapping, 10, 'hello worl', 20))

Listing 18.31: Complete example of generating characters with the fit model.
Running the example generates three sequences of text. The first is a test to see how the
model does at starting from the beginning of the rhyme. The second is a test to see how well it
does at beginning in the middle of a line. The final example is a test to see how well it does
with a sequence of characters never seen before.
Note: Given the stochastic nature of neural networks, your specific results may vary. Consider
running the example a few times.
Sing a song of sixpence, A poc
king was in his counting house
hello worls e pake wofey. The

Listing 18.32: Example output from generating sequences of characters.
We can see that the model did very well with the first two examples, as we would expect.
We can also see that the model still generated something for the new text, but it is nonsense.

18.6

Further Reading

This section provides more resources on the topic if you are looking go deeper.
 Sing a Song of Sixpence on Wikipedia.
https://en.wikipedia.org/wiki/Sing_a_Song_of_Sixpence
 Keras Utils API.
https://keras.io/utils/
 Keras Sequence Processing API.
https://keras.io/preprocessing/sequence/

18.7

Summary

In this tutorial, you discovered how to develop a character-based neural language model.
Specifically, you learned:
 How to prepare text for character-based language modeling.
 How to develop a character-based language model using LSTMs.
 How to use a trained character-based language model to generate text.

18.7.1

Next

In the next chapter, you will discover how you can develop a word-based neural language model.
