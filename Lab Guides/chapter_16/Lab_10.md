# Project: Develop an n-gram CNN Model for Sentiment Analysis
A standard deep learning model for text classification and sentiment analysis uses a word
embedding layer and one-dimensional convolutional neural network. The model can be expanded
by using multiple parallel convolutional neural networks that read the source document using
different kernel sizes. This, in effect, creates a multichannel convolutional neural network for
text that reads text with different n-gram sizes (groups of words). In this tutorial, you will
discover how to develop a multichannel convolutional neural network for sentiment prediction
on text movie review data. After completing this tutorial, you will know:
- How to prepare movie review text data for modeling.
- How to develop a multichannel convolutional neural network for text in Keras.
- How to evaluate a fit model on unseen movie review data.

Let's get started.

16.1

Tutorial Overview

This tutorial is divided into the following parts:
1. Movie Review Dataset.
2. Data Preparation.
3. Develop Multichannel Model.
4. Evaluate Model.

16.2

Movie Review Dataset

In this tutorial, we will use the Movie Review Dataset. This dataset designed for sentiment
analysis was described previously . You can download the dataset from here:

- Movie Review Polarity Dataset (review polarity.tar.gz, 3MB).
http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.
gz

After unzipping the file, you will have a directory called txt sentoken with two subdirectories containing the text neg and pos for negative and positive reviews. Reviews are stored
one per file with a naming convention cv000 to cv999 for each of neg and pos.

16.3

Data Preparation

Note: The preparation of the movie review dataset was first described previously. In this
section, we will look at 3 things:
1. Separation of data into training and test sets.
2. Loading and cleaning the data to remove punctuation and numbers.
3. Clean All Reviews and Save.

16.3.1

Split into Train and Test Sets

We are pretending that we are developing a system that can predict the sentiment of a textual
movie review as either positive or negative. This means that after the model is developed, we
will need to make predictions on new textual reviews. This will require all of the same data
preparation to be performed on those new reviews as is performed on the training data for the
model. We will ensure that this constraint is built into the evaluation of our models by splitting
the training and test datasets prior to any data preparation. This means that any knowledge in
the data in the test set that could help us better prepare the data (e.g. the words used) are
unavailable in the preparation of data used for training the model.
That being said, we will use the last 100 positive reviews and the last 100 negative reviews
as a test set (100 reviews) and the remaining 1,800 reviews as the training dataset. This is a
90% train, 10% split of the data. The split can be imposed easily by using the filenames of the
reviews where reviews named 000 to 899 are for training data and reviews named 900 onwards
are for test.

16.3.2

Loading and Cleaning Reviews

The text data is already pretty clean; not much preparation is required. Without getting bogged
down too much in the details, we will prepare the data using the following way:
- Split tokens on white space.
- Remove all punctuation from words.
- Remove all words that are not purely comprised of alphabetical characters.
- Remove all words that are known stop words.
- Remove all words that have a length ≤ 1 character.

16.3. Data Preparation

176

We can put all of these steps into a function called clean doc() that takes as an argument
the raw text loaded from a file and returns a list of cleaned tokens. We can also define a function
load doc() that loads a document from file ready for use with the clean doc() function. An
example of cleaning the first positive review is listed below.
from nltk.corpus import stopwords
import string
import re
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# turn a doc into clean tokens
def clean_doc(doc):
# split into tokens by white space
tokens = doc.split()
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# remove punctuation from each word
tokens = [re_punc.sub('', w) for w in tokens]
# remove remaining tokens that are not alphabetic
tokens = [word for word in tokens if word.isalpha()]
# filter out stop words
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
# filter out short tokens
tokens = [word for word in tokens if len(word) > 1]
return tokens
# load the document
filename = 'txt_sentoken/pos/cv000_29590.txt'
text = load_doc(filename)
tokens = clean_doc(text)
print(tokens)

```

Running the example prints a long list of clean tokens. There are many more cleaning steps
we may want to explore and I leave them as further exercises.
...
'creepy', 'place', 'even', 'acting', 'hell', 'solid', 'dreamy', 'depp', 'turning',
'typically', 'strong', 'performance', 'deftly', 'handling', 'british', 'accent',
'ians', 'holm', 'joe', 'goulds', 'secret', 'richardson', 'dalmatians', 'log', 'great',
'supporting', 'roles', 'big', 'surprise', 'graham', 'cringed', 'first', 'time',
'opened', 'mouth', 'imagining', 'attempt', 'irish', 'accent', 'actually', 'wasnt',
'half', 'bad', 'film', 'however', 'good', 'strong', 'violencegore', 'sexuality',
'language', 'drug', 'content']

```


16.3. Data Preparation

16.3.3

177

Clean All Reviews and Save

We can now use the function to clean reviews and apply it to all reviews. To do this, we will
develop a new function named process docs() below that will walk through all reviews in a
directory, clean them, and return them as a list. We will also add an argument to the function
to indicate whether the function is processing train or test reviews, that way the filenames can
be filtered (as described above) and only those train or test reviews requested will be cleaned
and returned. The full function is listed below.
# load all docs in a directory
def process_docs(directory, is_train):
documents = list()
# walk through all files in the folder
for filename in listdir(directory):
# skip any reviews in the test set
if is_train and filename.startswith('cv9'):
continue
if not is_train and not filename.startswith('cv9'):
continue
# create the full path of the file to open
path = directory + '/' + filename
# load the doc
doc = load_doc(path)
# clean doc
tokens = clean_doc(doc)
# add to list
documents.append(tokens)
return documents

```

We can call this function with negative training reviews. We also need labels for the train
and test documents. We know that we have 900 training documents and 100 test documents.
We can use a Python list comprehension to create the labels for the negative (0) and positive
(1) reviews for both train and test sets. The function below named load clean dataset() will
load and clean the movie review text and also create the labels for the reviews.
# load and clean a dataset
def load_clean_dataset(is_train):
# load documents
neg = process_docs('txt_sentoken/neg', is_train)
pos = process_docs('txt_sentoken/pos', is_train)
docs = neg + pos
# prepare labels
labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
return docs, labels

```

Finally, we want to save the prepared train and test sets to file so that we can load them
later for modeling and model evaluation. The function below-named save dataset() will save
a given prepared dataset (X and y elements) to a file using the pickle API (this is the standard
API for saving objects in Python).
# save a dataset to file
def save_dataset(dataset, filename):

16.3. Data Preparation

178

dump(dataset, open(filename, 'wb'))
print('Saved: %s' % filename)

```


16.3.4

Complete Example

We can tie all of these data preparation steps together. The complete example is listed below.
import string
import re
from os import listdir
from nltk.corpus import stopwords
from pickle import dump
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# turn a doc into clean tokens
def clean_doc(doc):
# split into tokens by white space
tokens = doc.split()
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# remove punctuation from each word
tokens = [re_punc.sub('', w) for w in tokens]
# remove remaining tokens that are not alphabetic
tokens = [word for word in tokens if word.isalpha()]
# filter out stop words
stop_words = set(stopwords.words('english'))
tokens = [w for w in tokens if not w in stop_words]
# filter out short tokens
tokens = [word for word in tokens if len(word) > 1]
tokens = ' '.join(tokens)
return tokens
# load all docs in a directory
def process_docs(directory, is_train):
documents = list()
# walk through all files in the folder
for filename in listdir(directory):
# skip any reviews in the test set
if is_train and filename.startswith('cv9'):
continue
if not is_train and not filename.startswith('cv9'):
continue
# create the full path of the file to open
path = directory + '/' + filename

16.4. Develop Multichannel Model

179

# load the doc
doc = load_doc(path)
# clean doc
tokens = clean_doc(doc)
# add to list
documents.append(tokens)
return documents
# load and clean a dataset
def load_clean_dataset(is_train):
# load documents
neg = process_docs('txt_sentoken/neg', is_train)
pos = process_docs('txt_sentoken/pos', is_train)
docs = neg + pos
# prepare labels
labels = [0 for _ in range(len(neg))] + [1 for _ in range(len(pos))]
return docs, labels
# save a dataset to file
def save_dataset(dataset, filename):
dump(dataset, open(filename, 'wb'))
print('Saved: %s' % filename)
# load and clean all reviews
train_docs, ytrain = load_clean_dataset(True)
test_docs, ytest = load_clean_dataset(False)
# save training datasets
save_dataset([train_docs, ytrain], 'train.pkl')
save_dataset([test_docs, ytest], 'test.pkl')

```

Running the example cleans the text movie review documents, creates labels, and saves the
prepared data for both train and test datasets in train.pkl and test.pkl respectively. Now
we are ready to develop our model.

16.4

Develop Multichannel Model

In this section, we will develop a multichannel convolutional neural network for the sentiment
analysis prediction problem. This section is divided into 3 parts:
1. Encode Data
2. Define Model.
3. Complete Example.

16.4.1

Encode Data

The first step is to load the cleaned training dataset. The function below-named load dataset()
can be called to load the pickled training dataset.

16.4. Develop Multichannel Model

180

# load a clean dataset
def load_dataset(filename):
return load(open(filename, 'rb'))
trainLines, trainLabels = load_dataset('train.pkl')

```

Next, we must fit a Keras Tokenizer on the training dataset. We will use this tokenizer to
both define the vocabulary for the Embedding layer and encode the review documents as integers.
The function create tokenizer() below will create a Tokenizer given a list of documents.
# fit a tokenizer
def create_tokenizer(lines):
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
return tokenizer

```

We also need to know the maximum length of input sequences as input for the model and
to pad all sequences to the fixed length. The function max length() below will calculate the
maximum length (number of words) for all reviews in the training dataset.
# calculate the maximum document length
def max_length(lines):
return max([len(s.split()) for s in lines])

```

We also need to know the size of the vocabulary for the Embedding layer. This can be
calculated from the prepared Tokenizer, as follows:
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1

```

Finally, we can integer encode and pad the clean movie review text. The function below
named encode text() will both encode and pad text data to the maximum review length.
# encode a list of lines
def encode_text(tokenizer, lines, length):
# integer encode
encoded = tokenizer.texts_to_sequences(lines)
# pad encoded sequences
padded = pad_sequences(encoded, maxlen=length, padding='post')
return padded

```


16.4.2

Define Model

A standard model for document classification is to use an Embedding layer as input, followed by
a one-dimensional convolutional neural network, pooling layer, and then a prediction output
layer. The kernel size in the convolutional layer defines the number of words to consider as

16.4. Develop Multichannel Model

181

the convolution is passed across the input text document, providing a grouping parameter. A
multi-channel convolutional neural network for document classification involves using multiple
versions of the standard model with different sized kernels. This allows the document to be
processed at different resolutions or different n-grams (groups of words) at a time, whilst the
model learns how to best integrate these interpretations.
This approach was first described by Yoon Kim in his 2014 paper titled Convolutional Neural
Networks for Sentence Classification. In the paper, Kim experimented with static and dynamic
(updated) embedding layers, we can simplify the approach and instead focus only on the use of
different kernel sizes. This approach is best understood with a diagram taken from Kim's paper.

In Keras, a multiple-input model can be defined using the functional API. We will define a
model with three input channels for processing 4-grams, 6-grams, and 8-grams of movie review
text. Each channel is comprised of the following elements:
- Input layer that defines the length of input sequences.
- Embedding layer set to the size of the vocabulary and 100-dimensional real-valued representations.
- Conv1D layer with 32 filters and a kernel size set to the number of words to read at once.
- MaxPooling1D layer to consolidate the output from the convolutional layer.
- Flatten layer to reduce the three-dimensional output to two dimensional for concatenation.

The output from the three channels are concatenated into a single vector and process by a
Dense layer and an output layer. The function below defines and returns the model. As part of
defining the model, a summary of the defined model is printed and a plot of the model graph is
created and saved to file.
# define the model
def define_model(length, vocab_size):
# channel 1
inputs1 = Input(shape=(length,))
embedding1 = Embedding(vocab_size, 100)(inputs1)
conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)
# channel 2
inputs2 = Input(shape=(length,))
embedding2 = Embedding(vocab_size, 100)(inputs2)
conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)
# channel 3
inputs3 = Input(shape=(length,))
embedding3 = Embedding(vocab_size, 100)(inputs3)
conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)

16.4. Develop Multichannel Model
# merge
merged = concatenate([flat1, flat2, flat3])
# interpretation
dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(1, activation='sigmoid')(dense1)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize
model.summary()
plot_model(model, show_shapes=True, to_file='multichannel.png')
return model

```


16.4.3

Complete Example

Pulling all of this together, the complete example is listed below.
from
from
from
from
from
from
from
from
from
from
from
from
from
from

pickle import load
numpy import array
keras.preprocessing.text import Tokenizer
keras.preprocessing.sequence import pad_sequences
keras.utils.vis_utils import plot_model
keras.models import Model
keras.layers import Input
keras.layers import Dense
keras.layers import Flatten
keras.layers import Dropout
keras.layers import Embedding
keras.layers.convolutional import Conv1D
keras.layers.convolutional import MaxPooling1D
keras.layers.merge import concatenate

# load a clean dataset
def load_dataset(filename):
return load(open(filename, 'rb'))
# fit a tokenizer
def create_tokenizer(lines):
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
return tokenizer
# calculate the maximum document length
def max_length(lines):
return max([len(s.split()) for s in lines])
# encode a list of lines
def encode_text(tokenizer, lines, length):
# integer encode
encoded = tokenizer.texts_to_sequences(lines)
# pad encoded sequences
padded = pad_sequences(encoded, maxlen=length, padding='post')
return padded

182

16.4. Develop Multichannel Model

# define the model
def define_model(length, vocab_size):
# channel 1
inputs1 = Input(shape=(length,))
embedding1 = Embedding(vocab_size, 100)(inputs1)
conv1 = Conv1D(filters=32, kernel_size=4, activation='relu')(embedding1)
drop1 = Dropout(0.5)(conv1)
pool1 = MaxPooling1D(pool_size=2)(drop1)
flat1 = Flatten()(pool1)
# channel 2
inputs2 = Input(shape=(length,))
embedding2 = Embedding(vocab_size, 100)(inputs2)
conv2 = Conv1D(filters=32, kernel_size=6, activation='relu')(embedding2)
drop2 = Dropout(0.5)(conv2)
pool2 = MaxPooling1D(pool_size=2)(drop2)
flat2 = Flatten()(pool2)
# channel 3
inputs3 = Input(shape=(length,))
embedding3 = Embedding(vocab_size, 100)(inputs3)
conv3 = Conv1D(filters=32, kernel_size=8, activation='relu')(embedding3)
drop3 = Dropout(0.5)(conv3)
pool3 = MaxPooling1D(pool_size=2)(drop3)
flat3 = Flatten()(pool3)
# merge
merged = concatenate([flat1, flat2, flat3])
# interpretation
dense1 = Dense(10, activation='relu')(merged)
outputs = Dense(1, activation='sigmoid')(dense1)
model = Model(inputs=[inputs1, inputs2, inputs3], outputs=outputs)
# compile
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# summarize
model.summary()
plot_model(model, show_shapes=True, to_file='model.png')
return model
# load training dataset
trainLines, trainLabels = load_dataset('train.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
print('Max document length: %d' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
# define model
model = define_model(length, vocab_size)
# fit model
model.fit([trainX,trainX,trainX], array(trainLabels), epochs=7, batch_size=16)
# save the model
model.save('model.h5')

183

16.4. Develop Multichannel Model

184

```

Running the example first prints a summary of the prepared training dataset.
Max document length: 1380
Vocabulary size: 44277

```

The model is fit relatively quickly and appears to show good skill on the training dataset.
...
Epoch 3/7
1800/1800
Epoch 4/7
1800/1800
Epoch 5/7
1800/1800
Epoch 6/7
1800/1800
Epoch 7/7
1800/1800

[==============================] - 29s - loss: 0.0460 - acc: 0.9894
[==============================] - 30s - loss: 0.0041 - acc: 1.0000
[==============================] - 31s - loss: 0.0010 - acc: 1.0000
[==============================] - 30s - loss: 3.0271e-04 - acc: 1.0000
[==============================] - 28s - loss: 1.3875e-04 - acc: 1.0000

```

A plot of the defined model is saved to file, clearly showing the three input channels for the
model.

![](./-.png)
The model is fit for a number of epochs and saved to the file model.h5 for later evaluation.

16.5. Evaluate Model

16.5

185

Evaluate Model

In this section, we can evaluate the fit model by predicting the sentiment on all reviews in the
unseen test dataset. Using the data loading functions developed in the previous section, we can
load and encode both the training and test datasets.
# load datasets
trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Max document length: %d' % length)
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
print(trainX.shape, testX.shape)

```

We can load the saved model and evaluate it on both the training and test datasets. The
complete example is listed below.

```
from pickle import load
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
# load a clean dataset
def load_dataset(filename):
return load(open(filename, 'rb'))
# fit a tokenizer
def create_tokenizer(lines):
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
return tokenizer
# calculate the maximum document length
def max_length(lines):
return max([len(s.split()) for s in lines])
# encode a list of lines
def encode_text(tokenizer, lines, length):
# integer encode
encoded = tokenizer.texts_to_sequences(lines)
# pad encoded sequences
padded = pad_sequences(encoded, maxlen=length, padding='post')
return padded
# load datasets
trainLines, trainLabels = load_dataset('train.pkl')
testLines, testLabels = load_dataset('test.pkl')
# create tokenizer
tokenizer = create_tokenizer(trainLines)
# calculate max document length
length = max_length(trainLines)
print('Max document length: %d' % length)
# calculate vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary size: %d' % vocab_size)
# encode data
trainX = encode_text(tokenizer, trainLines, length)
testX = encode_text(tokenizer, testLines, length)
# load the model
model = load_model('model.h5')
# evaluate model on training dataset
_, acc = model.evaluate([trainX,trainX,trainX], array(trainLabels), verbose=0)
print('Train Accuracy: %.2f' % (acc*100))
# evaluate model on test dataset dataset
_, acc = model.evaluate([testX,testX,testX], array(testLabels), verbose=0)
print('Test Accuracy: %.2f' % (acc*100))

```

Running the example prints the skill of the model on both the training and test datasets. We
can see that, as expected, the skill on the training dataset is excellent, here at 100% accuracy.
We can also see that the skill of the model on the unseen test dataset is also very impressive,
achieving 88.5%, which is above the skill of the model reported in the 2014 paper (although not
a direct apples-to-apples comparison).
Note: Given the stochastic nature of neural networks, your specific results may vary. Consider
running the example a few times.

```
Train Accuracy: 100.00
Test Accuracy: 88.50

```


# Extensions

This section lists some ideas for extending the tutorial that you may wish to explore.
- Different n-grams. Explore the model by changing the kernel size (number of n-grams)
used by the channels in the model to see how it impacts model skill.
- More or Fewer Channels. Explore using more or fewer channels in the model and see
how it impacts model skill.
- Shared Embedding. Explore configurations where each channel shares the same word
embedding and report on the impact on model skill.
- Deeper Network. Convolutional neural networks perform better in computer vision
when they are deeper. Explore using deeper models here and see how it impacts model
skill.
- Truncated Sequences. Padding all sequences to the length of the longest sequence
might be extreme if the longest sequence is very different to all other reviews. Study the
distribution of review lengths and truncate reviews to a mean length.
- Truncated Vocabulary. We removed infrequently occurring words, but still had a large
vocabulary of more than 25,000 words. Explore further reducing the size of the vocabulary
and the effect on model skill.
- Epochs and Batch Size. The model appears to fit the training dataset quickly. Explore
alternate configurations of the number of training epochs and batch size and use the test
dataset as a validation set to pick a better stopping point for training the model.
- Pre-Train an Embedding. Explore pre-training a Word2Vec word embedding in the
model and the impact on model skill with and without further fine tuning during training.
- Use GloVe Embedding. Explore loading the pre-trained GloVe embedding and the
impact on model skill with and without further fine tuning during training.
- Train Final Model. Train a final model on all available data and use it make predictions
on real ad hoc movie reviews from the internet.

If you explore any of these extensions, I'd love to know.

# Further Reading

This section provides more resources on the topic if you are looking to go deeper.
- Convolutional Neural Networks for Sentence Classification, 2014.
https://arxiv.org/abs/1408.5882
- Convolutional Neural Networks for Sentence Classification (code).
https://github.com/yoonkim/CNN_sentence
- Keras Functional API.
https://keras.io/getting-started/functional-api-guide/

# Summary

In this tutorial, you discovered how to develop a multichannel convolutional neural network for
sentiment prediction on text movie review data. Specifically, you learned:
- How to prepare movie review text data for modeling.
- How to develop a multichannel convolutional neural network for text in Keras.
- How to evaluate a fit model on unseen movie review data.

##### Run Notebook
Click notebook `1_clean_review.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `2_clean_all_reviews.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `3_model.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `4_evaluate.ipynb` in jupterLab UI and run jupyter notebook.

