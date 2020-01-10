
# How to Prepare Text Data With Keras
You cannot feed raw text directly into deep learning models. Text data must be encoded as
numbers to be used as input or output for machine learning and deep learning models, such
as word embeddings. The Keras deep learning library provides some basic tools to help you
prepare your text data. In this tutorial, you will discover how you can use Keras to prepare
your text data. After completing this tutorial, you will know:
- About the convenience methods that you can use to quickly prepare text data.
- The Tokenizer API that can be fit on training data and used to encode training, validation,
and test documents.
- The range of 4 different document encoding schemes offered by the Tokenizer API.

Let's get started.

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/deep-learning-for-nlp` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab4_Prepare_Text_Data_Keras`

# Tutorial Overview

This tutorial is divided into the following parts:
1. Split words with text to word sequence.
2. Encoding with one hot.
3. Hash Encoding with hashing trick.
4. Tokenizer API

### Split Words with text to word sequence

A good first step when working with text is to split it into words. Words are called tokens and the process of splitting text into tokens is called tokenization. Keras provides the
text to word sequence() function that you can use to split text into a list of words. By
default, this function automatically does 3 things:
- Splits words by space.
- Filters out punctuation.
- Converts text to lowercase (lower=True).

You can change any of these defaults by passing arguments to the function. Below is an
example of using the text to word sequence() function to split a document (in this case a
simple string) into a list of words.

```
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# tokenize the document
result = text_to_word_sequence(text)
print(result)

```

Running the example creates an array containing all of the words in the document. The list
of words is printed for review.

```
['the', 'quick', 'brown', 'fox', 'jumped', 'over', 'the', 'lazy', 'dog']

```

This is a good first step, but further pre-processing is required before you can work with the
text.

#### Encoding with one hot

It is popular to represent a document as a sequence of integer values, where each word in the
document is represented as a unique integer. Keras provides the one hot() function that you
can use to tokenize and integer encode a text document in one step. The name suggests that it
will create a one hot encoding of the document, which is not the case. Instead, the function
is a wrapper for the hashing trick() function described in the next section. The function
returns an integer encoded version of the document. The use of a hash function means that
there may be collisions and not all words will be assigned unique integer values. As with the
text to word sequence() function in the previous section, the one hot() function will make
the text lower case, filter out punctuation, and split words based on white space.
In addition to the text, the vocabulary size (total words) must be specified. This could be the
total number of words in the document or more if you intend to encode additional documents
that contains additional words. The size of the vocabulary defines the hashing space from which
words are hashed. By default, the hash function is used, although as we will see in the next
section, alternate hash functions can be specified when calling the hashing trick() function
directly.

We can use the text to word sequence() function from the previous section to split the
document into words and then use a set to represent only the unique words in the document.
The size of this set can be used to estimate the size of the vocabulary for one document. For
example:

```
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)

```

We can put this together with the one hot() function and encode the words in the document.
The complete example is listed below. The vocabulary size is increased by one-third to minimize
collisions when hashing words.

```
from keras.preprocessing.text import one_hot
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'
# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
# integer encode the document
result = one_hot(text, round(vocab_size*1.3))
print(result)

```

Running the example first prints the size of the vocabulary as 8. The encoded document is
then printed as an array of integer encoded words.
**Note:**  Given the stochastic nature of neural networks, your specific results may vary. Consider
running the example a few times.
```
8
[5, 9, 8, 7, 9, 1, 5, 3, 8]

```


# Hash Encoding with hashing trick

A limitation of integer and count base encodings is that they must maintain a vocabulary of
words and their mapping to integers. An alternative to this approach is to use a one-way hash
function to convert words to integers. This avoids the need to keep track of a vocabulary, which
is faster and requires less memory.
Keras provides the hashing trick() function that tokenizes and then integer encodes the
document, just like the one hot() function. It provides more flexibility, allowing you to specify
the hash function as either hash (the default) or other hash functions such as the built in md5
function or your own function. Below is an example of integer encoding a document using the
md5 hash function.

```
from keras.preprocessing.text import hashing_trick
from keras.preprocessing.text import text_to_word_sequence
# define the document
text = 'The quick brown fox jumped over the lazy dog.'

# estimate the size of the vocabulary
words = set(text_to_word_sequence(text))
vocab_size = len(words)
print(vocab_size)
# integer encode the document
result = hashing_trick(text, round(vocab_size*1.3), hash_function='md5')
print(result)

```

Running the example prints the size of the vocabulary and the integer encoded document.
We can see that the use of a different hash function results in consistent, but different integers
for words as the one hot() function in the previous section.

```
8
[6, 4, 1, 2, 7, 5, 6, 2, 6]

```

## Tokenizer API

So far we have looked at one-off convenience methods for preparing text with Keras. Keras
provides a more sophisticated API for preparing text that can be fit and reused to prepare
multiple text documents. This may be the preferred approach for large projects. Keras provides
the Tokenizer class for preparing text documents for deep learning. The Tokenizer must be
constructed and then fit on either raw text documents or integer encoded text documents. For
example:

```
from keras.preprocessing.text import Tokenizer
# define 5 documents
docs = ['Well done!',
'Good work',
'Great effort',
'nice work',
'Excellent!']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)

```

Once fit, the Tokenizer provides 4 attributes that you can use to query what has been
learned about your documents:
- word counts: A dictionary mapping of words and their occurrence counts when the
Tokenizer was fit.
- word docs: A dictionary mapping of words and the number of documents that reach
appears in.
- word index: A dictionary of words and their uniquely assigned integers.
- document count: A dictionary mapping and the number of documents they appear in
calculated during the fit.

For example:

```
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)

```

Once the Tokenizer has been fit on training data, it can be used to encode documents in
the train or test datasets. The texts to matrix() function on the Tokenizer can be used to
create one vector per document provided per input. The length of the vectors is the total size
of the vocabulary. This function provides a suite of standard bag-of-words model text encoding
schemes that can be provided via a mode argument to the function. The modes available
include:
- binary: Whether or not each word is present in the document. This is the default.
- count: The count of each word in the document.
- tfidf: The Text Frequency-Inverse DocumentFrequency (TF-IDF) scoring for each word
in the document.
- freq: The frequency of each word as a ratio of words within each document.

We can put all of this together with a worked example.

```
from keras.preprocessing.text import Tokenizer
# define 5 documents
docs = ['Well done!',
'Good work',
'Great effort',
'nice work',
'Excellent!']
# create the tokenizer
t = Tokenizer()
# fit the tokenizer on the documents
t.fit_on_texts(docs)
# summarize what was learned
print(t.word_counts)
print(t.document_count)
print(t.word_index)
print(t.word_docs)
# integer encode documents
encoded_docs = t.texts_to_matrix(docs, mode='count')
print(encoded_docs)

```

Running the example fits the Tokenizer with 5 small documents. The details of the fit
Tokenizer are printed. Then the 5 documents are encoded using a word count. Each document
is encoded as a 9-element vector with one position for each word and the chosen encoding
scheme value for each word position. In this case, a simple word count mode is used.

```
OrderedDict([('well', 1), ('done', 1), ('good', 1), ('work', 2), ('great', 1), ('effort',
1), ('nice', 1), ('excellent', 1)])
5
{'work': 1, 'effort': 6, 'done': 3, 'great': 5, 'good': 4, 'excellent': 8, 'well': 2,
'nice': 7}
{'work': 2, 'effort': 1, 'done': 1, 'well': 1, 'good': 1, 'great': 1, 'excellent': 1,
'nice': 1}
[[ 0. 0. 1. 1. 0. 0. 0. 0. 0.]
[ 0. 1. 0. 0. 1. 0. 0. 0. 0.]
[ 0. 0. 0. 0. 0. 1. 1. 0. 0.]
[ 0. 1. 0. 0. 0. 0. 0. 1. 0.]
[ 0. 0. 0. 0. 0. 0. 0. 0. 1.]]

```

The Tokenizer will be the key way we will prepare text for word embeddings throughout
this course.

##### Run Notebook
Click notebook `1_split_words.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `2_prepare_vocab.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `3_one_hot_encode.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `4_hash_encode.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `5_example_tokenizer.ipynb` in jupterLab UI and run jupyter notebook.

# Further Reading

This section provides more resources on the topic if you are looking go deeper.
- Text Preprocessing Keras API.
https://keras.io/preprocessing/text/
- text to word sequence Keras API.
https://keras.io/preprocessing/text/#text_to_word_sequence
- one hot Keras API.
https://keras.io/preprocessing/text/#one_hot
- hashing trick Keras API.
https://keras.io/preprocessing/text/#hashing_trick
- Tokenizer Keras API.
https://keras.io/preprocessing/text/#tokenizer

# Summary

In this tutorial, you discovered how you can use the Keras API to prepare your text data for
deep learning. Specifically, you learned:
- About the convenience methods that you can use to quickly prepare text data.
- The Tokenizer API that can be fit on training data and used to encode training, validation,
and test documents.
- The range of 4 different document encoding schemes offered by the Tokenizer API.
