# How to Learn and Load Word Embeddings in Keras
Word embeddings provide a dense representation of words and their relative meanings. They are
an improvement over sparse representations used in simpler bag of word model representations.
Word embeddings can be learned from text data and reused among projects. They can also be
learned as part of fitting a neural network on text data. In this tutorial, you will discover how
to use word embeddings for deep learning in Python with Keras. After completing this tutorial,
you will know:
- About word embeddings and that Keras supports word embeddings via the Embedding
layer.
- How to learn a word embedding while fitting a neural network.
- How to use a pre-trained word embedding in a neural network.

Let’s get started.

# Tutorial Overview

This tutorial is divided into the following parts:
1. Word Embedding
2. Keras Embedding Layer
3. Example of Learning an Embedding
4. Example of Using Pre-Trained GloVe Embedding
5. Tips for Cleaning Text for Word Embedding

Word Embedding

A word embedding is a class of approaches for representing words and documents using a
dense vector representation. It is an improvement over more the traditional bag-of-word model
encoding schemes where large sparse vectors were used to represent each word or to score each
word within a vector to represent an entire vocabulary. These representations were sparse
because the vocabularies were vast and a given word or document would be represented by a
large vector comprised mostly of zero values.
Instead, in an embedding, words are represented by dense vectors where a vector represents
the projection of the word into a continuous vector space. The position of a word within the
vector space is learned from text and is based on the words that surround the word when it is
used. The position of a word in the learned vector space is referred to as its embedding. Two
popular examples of methods of learning word embeddings from text include:
- Word2Vec.
- GloVe.

In addition to these carefully designed methods, a word embedding can be learned as part
of a deep learning model. This can be a slower approach, but tailors the model to a specific
training dataset.

13.3

Keras Embedding Layer

Keras offers an Embedding layer that can be used for neural networks on text data. It requires
that the input data be integer encoded, so that each word is represented by a unique integer.
This data preparation step can be performed using the Tokenizer API also provided with
Keras.
The Embedding layer is initialized with random weights and will learn an embedding for all
of the words in the training dataset. It is a flexible layer that can be used in a variety of ways,
such as:
- It can be used alone to learn a word embedding that can be saved and used in another
model later.
- It can be used as part of a deep learning model where the embedding is learned along
with the model itself.
- It can be used to load a pre-trained word embedding model, a type of transfer learning.

The Embedding layer is defined as the first hidden layer of a network. It must specify 3
arguments:
- input dim: This is the size of the vocabulary in the text data. For example, if your data
is integer encoded to values between 0-10, then the size of the vocabulary would be 11
words.
- output dim: This is the size of the vector space in which words will be embedded. It
defines the size of the output vectors from this layer for each word. For example, it could
be 32 or 100 or even larger. Test different values for your problem.
- input length: This is the length of input sequences, as you would define for any input
layer of a Keras model. For example, if all of your input documents are comprised of 1000
words, this would be 1000.

For example, below we define an Embedding layer with a vocabulary of 200 (e.g. integer
encoded words from 0 to 199, inclusive), a vector space of 32 dimensions in which words will be
embedded, and input documents that have 50 words each.

```
e = Embedding(200, 32, input_length=50)
```

The Embedding layer has weights that are learned. If you save your model to file, this will
include weights for the Embedding layer. The output of the Embedding layer is a 2D vector with
one embedding for each word in the input sequence of words (input document). If you wish
to connect a Dense layer directly to an Embedding layer, you must first flatten the 2D output
matrix to a 1D vector using the Flatten layer. Now, let’s see how we can use an Embedding
layer in practice.

### Example of Learning an Embedding

In this section, we will look at how we can learn a word embedding while fitting a neural
network on a text classification problem. We will define a small problem where we have 10
text documents, each with a comment about a piece of work a student submitted. Each text
document is classified as positive 1 or negative 0. This is a simple sentiment analysis problem.
First, we will define the documents and their class labels.

```
# define documents
docs = ['Well done!',
'Good work',
'Great effort',
'nice work',
'Excellent!',
'Weak',
'Poor effort!',
'not good',
'poor work',
'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])

```

Next, we can integer encode each document. This means that as input the Embedding layer
will have sequences of integers. We could experiment with other more sophisticated bag of word
model encoding like counts or TF-IDF. Keras provides the one hot() function that creates a
hash of each word as an efficient integer encoding. We will estimate the vocabulary size of 50,
which is much larger than needed to reduce the probability of collisions from the hash function.

```
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)

```

The sequences have different lengths and Keras prefers inputs to be vectorized and all inputs
to have the same length. We will pad all input sequences to have the length of 4. Again, we can
do this with a built in Keras function, in this case the pad sequences() function.

```
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

```

We are now ready to define our Embedding layer as part of our neural network model.
The Embedding layer has a vocabulary of 50 and an input length of 4. We will choose a
small embedding space of 8 dimensions. The model is a simple binary classification model.
Importantly, the output from the Embedding layer will be 4 vectors of 8 dimensions each, one
for each word. We flatten this to a one 32-element vector to pass on to the Dense output layer.

```
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
model.summary()

```

Finally, we can fit and evaluate the classification model.

```
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

```

The complete code listing is provided below.

```
from numpy import array
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
# define documents
docs = ['Well done!',
'Good work',
'Great effort',
'nice work',
'Excellent!',
'Weak',
'Poor effort!',
'not good',
'poor work',
'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# integer encode the documents
vocab_size = 50
encoded_docs = [one_hot(d, vocab_size) for d in docs]
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# define the model
model = Sequential()
model.add(Embedding(vocab_size, 8, input_length=max_length))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
model.summary()
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

```

Running the example first prints the integer encoded documents.
```
[[6, 16], [42, 24], [2, 17], [42, 24], [18], [17], [22, 17], [27, 42], [22, 24], [49, 46,
16, 34]]

```

Then the padded versions of each document are printed, making them all uniform length.

```
[[ 6 16 0 0]
[42 24 0 0]
[ 2 17 0 0]
[42 24 0 0]
[18 0 0 0]
[17 0 0 0]
[22 17 0 0]
[27 42 0 0]
[22 24 0 0]
[49 46 16 34]]
```

After the network is defined, a summary of the layers is printed. We can see that as expected,
the output of the Embedding layer is a 4 × 8 matrix and this is squashed to a 32-element vector
by the Flatten layer.

```
_________________________________________________________________
Layer (type)
Output Shape
Param #
=================================================================
embedding_1 (Embedding)
(None, 4, 8)
400
_________________________________________________________________
flatten_1 (Flatten)
(None, 32)
0
_________________________________________________________________
dense_1 (Dense)
(None, 1)
33
=================================================================
Total params: 433
Trainable params: 433
Non-trainable params: 0
_________________________________________________________________

```

Finally, the accuracy of the trained model is printed, showing that it learned the training
dataset perfectly (which is not surprising).
**Note:**  Given the stochastic nature of neural networks, your specific results may vary. Consider
running the example a few times.

```
Accuracy: 100.000000

```

You could save the learned weights from the Embedding layer to file for later use in other
models. You could also use this model generally to classify other documents that have the
same kind vocabulary seen in the test dataset. Next, let’s look at loading a pre-trained word
embedding in Keras.

# Example of Using Pre-Trained GloVe Embedding

The Keras Embedding layer can also use a word embedding learned elsewhere. It is common
in the field of Natural Language Processing to learn, save, and make freely available word
embeddings. For example, the researchers behind GloVe method provide a suite of pre-trained
word embeddings on their website released under a public domain license.
The smallest package of embeddings is 822 Megabytes, called glove.6B.zip. 

#### Download Dataset
Dataset is very huge. Before running the notebook, **download** the dataset and unzip it.
`curl -L  http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip -o glove.6B.zip`

`unzip glove.6B.zip`

It was trained on a dataset of one billion tokens (words) with a vocabulary of 400 thousand words. There
are a few different embedding vector sizes, including 50, 100, 200 and 300 dimensions. You
can download this collection of embeddings and we can seed the Keras Embedding layer with
weights from the pre-trained embedding for the words in your training dataset.
This example is inspired by an example in the Keras project: pretrained word embeddings.py.
After downloading and unzipping, you will see a few files, one of which is glove.6B.100d.txt,
which contains a 100-dimensional version of the embedding. If you peek inside the file, you will
see a token (word) followed by the weights (100 numbers) on each line. For example, below are
the first line of the embedding ASCII text file showing the embedding for the.

```
the -0.038194 -0.24487 0.72812 -0.39961 0.083172 0.043953 -0.39141 0.3344 -0.57545 0.087459
0.28787 -0.06731 0.30906 -0.26384 -0.13231 -0.20757 0.33395 -0.33848 -0.31743 -0.48336
0.1464 -0.37304 0.34577 0.052041 0.44946 -0.46971 0.02628 -0.54155 -0.15518 -0.14107
-0.039722 0.28277 0.14393 0.23464 -0.31021 0.086173 0.20397 0.52624 0.17164 -0.082378
-0.71787 -0.41531 0.20335 -0.12763 0.41367 0.55187 0.57908 -0.33477 -0.36559 -0.54857
-0.062892 0.26584 0.30205 0.99775 -0.80481 -3.0243 0.01254 -0.36942 2.2167 0.72201
-0.24978 0.92136 0.034514 0.46745 1.1079 -0.19358 -0.074575 0.23353 -0.052062 -0.22044
0.057162 -0.15806 -0.30798 -0.41625 0.37972 0.15006 -0.53212 -0.2055 -1.2526 0.071624
0.70565 0.49744 -0.42063 0.26148 -1.538 -0.30223 -0.073438 -0.28312 0.37104 -0.25217
0.016215 -0.017099 -0.38984 0.87424 -0.72569 -0.51058 -0.52028 -0.1459 0.8278 0.27062

```

As in the previous section, the first step is to define the examples, encode them as integers,
then pad the sequences to be the same length. In this case, we need to be able to map words to
integers as well as integers to words. Keras provides a Tokenizer class that can be fit on the
training data, can convert text to sequences consistently by calling the texts to sequences()
method on the Tokenizer class, and provides access to the dictionary mapping of words to
integers in a word index attribute.

```
# define documents
docs = ['Well done!',
'Good work',
'Great effort',
'nice work',
'Excellent!',
'Weak',
'Poor effort!',
'not good',
'poor work',
'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)

```

Next, we need to load the entire GloVe word embedding file into memory as a dictionary of
word to embedding array.

```
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt')
for line in f:
values = line.split()
word = values[0]
coefs = asarray(values[1:], dtype='float32')
embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))

```

This is pretty slow. It might be better to filter the embedding for the unique words in your
training data. Next, we need to create a matrix of one embedding for each word in the training
dataset. We can do that by enumerating all unique words in the Tokenizer.word index and
locating the embedding weight vector from the loaded GloVe embedding. The result is a matrix
of weights only for words we will see during training.

```
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
embedding_vector = embeddings_index.get(word)
if embedding_vector is not None:
embedding_matrix[i] = embedding_vector

```

Now we can define our model, fit, and evaluate it as before. The key difference is that
the Embedding layer can be seeded with the GloVe word embedding weights. We chose the
100-dimensional version, therefore the Embedding layer must be defined with output dim set to
100. Finally, we do not want to update the learned word weights in this model, therefore we
will set the trainable attribute for the model to be False.

```
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
```

The complete worked example is listed below.

```
from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
# define documents
docs = ['Well done!',
'Good work',
'Great effort',
'nice work',
'Excellent!',
'Weak',
'Poor effort!',
'not good',
'poor work',
'Could have done better.']
# define class labels
labels = array([1,1,1,1,1,0,0,0,0,0])
# prepare tokenizer
t = Tokenizer()
t.fit_on_texts(docs)
vocab_size = len(t.word_index) + 1
# integer encode the documents
encoded_docs = t.texts_to_sequences(docs)
print(encoded_docs)
# pad documents to a max length of 4 words
max_length = 4
padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')
print(padded_docs)
# load the whole embedding into memory
embeddings_index = dict()
f = open('glove.6B.100d.txt', mode='rt', encoding='utf-8')
for line in f:
values = line.split()
word = values[0]
coefs = asarray(values[1:], dtype='float32')
embeddings_index[word] = coefs
f.close()
print('Loaded %s word vectors.' % len(embeddings_index))
# create a weight matrix for words in training docs
embedding_matrix = zeros((vocab_size, 100))
for word, i in t.word_index.items():
embedding_vector = embeddings_index.get(word)
if embedding_vector is not None:
embedding_matrix[i] = embedding_vector
# define model
model = Sequential()
e = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=4, trainable=False)
model.add(e)
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
# summarize the model
model.summary()
# fit the model
model.fit(padded_docs, labels, epochs=50, verbose=0)
# evaluate the model
loss, accuracy = model.evaluate(padded_docs, labels, verbose=0)
print('Accuracy: %f' % (accuracy*100))

```

Running the example may take a bit longer, but then demonstrates that it is just as capable
of fitting this simple problem.

**Note:**  Given the stochastic nature of neural networks, your specific results may vary. Consider
running the example a few times.

```
...
Accuracy: 100.000000
```

In practice, I would encourage you to experiment with learning a word embedding using
a pre-trained embedding that is fixed and trying to perform learning on top of a pre-trained
embedding. See what works best for your specific problem.


##### Run Notebook
Click notebook `1_embedding_example.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `2_pretrained_embedding.ipynb` in jupterLab UI and run jupyter notebook.


# Tips for Cleaning Text for Word Embedding

Recently, the field of natural language processing has been moving away from bag-of-word
models and word encoding toward word embeddings. The benefit of word embeddings is that
they encode each word into a dense vector that captures something about its relative meaning
within the training text. This means that variations of words like case, spelling, punctuation,
and so on will automatically be learned to be similar in the embedding space. In turn, this
can mean that the amount of cleaning required from your text may be less and perhaps quite
different to classical text cleaning. For example, it may no-longer make sense to stem words or
remove punctuation for contractions.
Tomas Mikolov is one of the developers of Word2Vec, a popular word embedding method.
He suggests only very minimal text cleaning is required when learning a word embedding model.
Below is his response when pressed with the question about how to best prepare text data for
Word2Vec.

There is no universal answer. It all depends on what you plan to use the vectors
for. In my experience, it is usually good to disconnect (or remove) punctuation from
words, and sometimes also convert all characters to lowercase. One can also replace
all numbers (possibly greater than some constant) with some single token such as .
All these pre-processing steps aim to reduce the vocabulary size without removing
any important content (which in some cases may not be true when you lowercase
certain words, ie. ‘Bush’ is different than ‘bush’, while ‘Another’ has usually the
same sense as ‘another’). The smaller the vocabulary is, the lower is the memory
complexity, and the more robustly are the parameters for the words estimated. You
also have to pre-process the test data in the same way.
[...]
In short, you will understand all this much better if you will run experiments.
— Tomas Mikolov, word2vec-toolkit: google groups thread., 2015.
https://goo.gl/KtDGst

# Further Reading

This section provides more resources on the topic if you are looking go deeper.
- Word Embedding on Wikipedia.
https://en.wikipedia.org/wiki/Word_embedding
- Keras Embedding Layer API.
https://keras.io/layers/embeddings/#embedding

# Summary

- Using pre-trained word embeddings in a Keras model, 2016.
https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
- Example of using a pre-trained GloVe Embedding in Keras.
https://github.com/fchollet/keras/blob/master/examples/pretrained_word_embeddings.
py
- GloVe Embedding.
https://nlp.stanford.edu/projects/glove/
- An overview of word embeddings and their connection to distributional semantic models,
2016.
http://blog.aylien.com/overview-word-embeddings-history-word2vec-cbow-glove/
- Deep Learning, NLP, and Representations, 2014.
http://colah.github.io/posts/2014-07-NLP-RNNs-Representations/

# Summary

In this tutorial, you discovered how to use word embeddings for deep learning in Python with
Keras. Specifically, you learned:
- About word embeddings and that Keras supports word embeddings via the Embedding
layer.
- How to learn a word embedding while fitting a neural network.
- How to use a pre-trained word embedding in a neural network.
