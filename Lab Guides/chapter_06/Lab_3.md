<img align="right" src="../logo-small.png">



# How to Prepare Text Data with scikit-learn

Text data requires special preparation before you can start using it for predictive modeling. The
text must be parsed to remove words, called tokenization. Then the words need to be encoded
as integers or floating point values for use as input to a machine learning algorithm, called
feature extraction (or vectorization). The scikit-learn library offers easy-to-use tools to perform
both tokenization and feature extraction of your text data. In this tutorial, you will discover
exactly how you can prepare your text data for predictive modeling in Python with scikit-learn.
After completing this tutorial, you will know:
- How to convert text to word count vectors with CountVectorizer.
- How to convert text to word frequency vectors with TfidfVectorizer.
- How to convert text to unique integers with HashingVectorizer.

Let's get started.

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/deep-learning-for-nlp` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab3_Prepare_Text_Data_scikit_learn`

## The Bag-of-Words Model

We cannot work with text directly when using machine learning algorithms. Instead, we need
to convert the text to numbers. We may want to perform classification of documents, so each
document is an input and a class label is the output for our predictive algorithm. Algorithms
take vectors of numbers as input, therefore we need to convert documents to fixed-length vectors
of numbers.
A simple and effective model for thinking about text documents in machine learning is called
the Bag-of-Words Model, or BoW. Note, that we cover the BoW model in great detail in the
next part, starting with Chapter 8. The model is simple in that it throws away all of the order
information in the words and focuses on the occurrence of words in a document. This can be
done by assigning each word a unique number. Then any document we see can be encoded
as a fixed-length vector with the length of the vocabulary of known words. The value in each
position in the vector could be filled with a count or frequency of each word in the encoded
document.

This is the bag-of-words model, where we are only concerned with encoding schemes that
represent what words are present or the degree to which they are present in encoded documents
without any information about order. There are many ways to extend this simple method, both
by better clarifying what a word is and in defining what to encode about each word in the
vector. The scikit-learn library provides 3 different schemes that we can use, and we will briefly
look at each.

#### Word Counts with CountVectorizer

The CountVectorizer provides a simple way to both tokenize a collection of text documents
and build a vocabulary of known words, but also to encode new documents using that vocabulary.
You can use it as follows:
- Create an instance of the CountVectorizer class.
- Call the fit() function in order to learn a vocabulary from one or more documents.
- Call the transform() function on one or more documents as needed to encode each as a
vector.

An encoded vector is returned with a length of the entire vocabulary and an integer count
for the number of times each word appeared in the document. Because these vectors will
contain a lot of zeros, we call them sparse. Python provides an efficient way of handling sparse
vectors in the scipy.sparse package. The vectors returned from a call to transform() will
be sparse vectors, and you can transform them back to NumPy arrays to look and better
understand what is going on by calling the toarray() function. Below is an example of using
the CountVectorizer to tokenize, build a vocabulary, and then encode a document.
from sklearn.feature_extraction.text import CountVectorizer

```
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = CountVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(type(vector))
print(vector.toarray())

```

Above, you can see that we access the vocabulary to see what exactly was tokenized by
calling:

```
print(vectorizer.vocabulary_)
```

We can see that all words were made lowercase by default and that the punctuation was
ignored. These and other aspects of tokenizing can be configured and I encourage you to review
all of the options in the API documentation. Running the example first prints the vocabulary,
then the shape of the encoded document. We can see that there are 8 words in the vocab, and
therefore encoded vectors have a length of 8. We can then see that the encoded vector is a
sparse matrix. Finally, we can see an array version of the encoded vector showing a count of 1
occurrence for each word except the (index and id 7) that has an occurrence of 2.

```
{'dog': 1, 'fox': 2, 'over': 5, 'brown': 0, 'quick': 6, 'the': 7, 'lazy': 4, 'jumped': 3}
(1, 8)
<class 'scipy.sparse.csr.csr_matrix'>
[[1 1 1 1 1 1 1 2]]

```

Importantly, the same vectorizer can be used on documents that contain words not included
in the vocabulary. These words are ignored and no count is given in the resulting vector. For
example, below is an example of using the vectorizer above to encode a document with one
word in the vocab and one word that is not.

```
# encode another document
text2 = ["the puppy"]
vector = vectorizer.transform(text2)
print(vector.toarray())

```


##### Run Notebook
Click notebook `1_example_countvector.ipynb` in jupterLab UI and run jupyter notebook.


Running this example prints the array version of the encoded sparse vector showing one
occurrence of the one word in the vocab and the other word not in the vocab completely ignored.

```
[[0 0 0 0 0 0 0 1]]

```

The encoded vectors can then be used directly with a machine learning algorithm.

# Word Frequencies with TfidfVectorizer

Word counts are a good starting point, but are very basic. One issue with simple counts is that
some words like the will appear many times and their large counts will not be very meaningful
in the encoded vectors. An alternative is to calculate word frequencies, and by far the most
popular method is called TF-IDF. This is an acronym that stands for Term Frequency - Inverse
Document Frequency which are the components of the resulting scores assigned to each word.
- Term Frequency: This summarizes how often a given word appears within a document.
- Inverse Document Frequency: This downscales words that appear a lot across documents.

Without going into the math, TF-IDF are word frequency scores that try to highlight
words that are more interesting, e.g. frequent in a document but not across documents.
The TfidfVectorizer will tokenize documents, learn the vocabulary and inverse document
frequency weightings, and allow you to encode new documents. Alternately, if you already have a
learned CountVectorizer, you can use it with a TfidfTransformer to just calculate the inverse
document frequencies and start encoding documents. The same create, fit, and transform process
is used as with the CountVectorizer. Below is an example of using the TfidfVectorizer to
learn vocabulary and inverse document frequencies across 3 small documents and then encode
one of those documents.

```
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
"The dog.",
"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())

```

A vocabulary of 8 words is learned from the documents and each word is assigned a unique
integer index in the output vector. The inverse document frequencies are calculated for each
word in the vocabulary, assigning the lowest score of 1.0 to the most frequently observed word:
the at index 7. Finally, the first document is encoded as an 8-element sparse array and we can
review the final scorings of each word with different values for the, fox, and dog from the other
words in the vocabulary.

```
{'fox': 2, 'lazy': 4, 'dog': 1, 'quick': 6, 'the': 7, 'over': 5, 'brown': 0, 'jumped': 3}
[ 1.69314718 1.28768207 1.28768207 1.69314718 1.69314718 1.69314718
1.69314718 1. ]
(1, 8)
[[ 0.36388646 0.27674503 0.27674503 0.36388646 0.36388646 0.36388646
0.36388646 0.42983441]]

```

##### Run Notebook
Click notebook `2_example_tfidf.ipynb` in jupterLab UI and run jupyter notebook.

The scores are normalized to values between 0 and 1 and the encoded document vectors can
then be used directly with most machine learning algorithms.

# Hashing with HashingVectorizer

Counts and frequencies can be very useful, but one limitation of these methods is that the
vocabulary can become very large. This, in turn, will require large vectors for encoding
documents and impose large requirements on memory and slow down algorithms. A clever work
around is to use a one way hash of words to convert them to integers. The clever part is that
no vocabulary is required and you can choose an arbitrary-long fixed length vector. A downside
is that the hash is a one-way function so there is no way to convert the encoding back to a word
(which may not matter for many supervised learning tasks).
The HashingVectorizer class implements this approach that can be used to consistently
hash words, then tokenize and encode documents as needed. The example below demonstrates
the HashingVectorizer for encoding a single document. An arbitrary fixed-length vector size
of 20 was chosen. This corresponds to the range of the hash function, where small values (like
20) may result in hash collisions. Remembering back to Computer Science classes, I believe
there are heuristics that you can use to pick the hash length and probability of collision based
on estimated vocabulary size (e.g. a load factor of 75%). See any good textbook on the topic.
Note that this vectorizer does not require a call to fit on the training data documents. Instead,
after instantiation, it can be used directly to start encoding documents.

```
from sklearn.feature_extraction.text import HashingVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog."]
# create the transform
vectorizer = HashingVectorizer(n_features=20)
# encode document
vector = vectorizer.transform(text)
# summarize encoded vector
print(vector.shape)
print(vector.toarray())

```

Running the example encodes the sample document as a 20-element sparse array. The values
of the encoded document correspond to normalized word counts by default in the range of -1 to
1, but could be made simple integer counts by changing the default configuration.

```
(1, 20)
[[ 0. 0. 0. 0. 0. 0.33333333
0. -0.33333333 0.33333333 0. 0. 0.33333333
0. 0. 0. -0.33333333 0. 0.
-0.66666667 0. ]]

```

##### Run Notebook
Click notebook `3_example_hash.ipynb` in jupterLab UI and run jupyter notebook.

# Further Reading

This section provides more resources on the topic if you are looking go deeper.


Natural Language Processing

- Bag-of-words model on Wikipedia.
https://en.wikipedia.org/wiki/Bag-of-words_model
- Tokenization on Wikipedia.
https://en.wikipedia.org/wiki/Lexical_analysis#Tokenization
- TF-IDF on Wikipedia.
https://en.wikipedia.org/wiki/Tf%E2%80%93idf
- Section 4.2. Feature extraction, scikit-learn User Guide.
http://scikit-learn.org/stable/modules/feature_extraction.html
- sckit-learn Feature Extraction API.
http://scikit-learn.org/stable/modules/classes.html#module-sklearn.feature_
extraction
- Working With Text Data, scikit-learn Tutorial.
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.
html

Class APIs

- CountVectorizer scikit-learn API.
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.
text.CountVectorizer.html
- TfidfVectorizer scikit-learn API.
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.
text.TfidfVectorizer.html
- TfidfTransformer scikit-learn API.
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.
text.TfidfTransformer.html
- HashingVectorizer scikit-learn API.
http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.
text.HashingVectorizer.html

# Summary

In this tutorial, you discovered how to prepare text documents for machine learning with
scikit-learn for bag-of-words models. Specifically, you learned:
- How to convert text to word count vectors with CountVectorizer.
- How to convert text to word frequency vectors with TfidfVectorizer.
- How to convert text to unique integers with HashingVectorizer.

We have only scratched the surface in these examples and I want to highlight that there are
many configuration details for these classes to influence the tokenizing of documents that are
worth exploring.

# Next

In the next lab, you will discover how you can prepare text data using the Keras deep
learning library.
