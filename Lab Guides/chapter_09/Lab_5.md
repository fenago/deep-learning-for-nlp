
# How to Prepare Movie Review Data for Sentiment Analysis
Text data preparation is different for each problem. Preparation starts with simple steps, like
loading data, but quickly gets difficult with cleaning tasks that are very specific to the data you
are working with. You need help as to where to begin and what order to work through the steps
from raw data to data ready for modeling. In this tutorial, you will discover how to prepare
movie review text data for sentiment analysis, step-by-step. After completing this tutorial, you
will know:
- How to load text data and clean it to remove punctuation and other non-words.
- How to develop a vocabulary, tailor it, and save it to file.
- How to prepare movie reviews using cleaning and a pre-defined vocabulary and save them
to new files ready for modeling.

Let's get started.

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/deep-learning-for-nlp` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab5_Movie_Review_Sentiment_Analysis`


# Tutorial Overview

This tutorial is divided into the following parts:
1. Movie Review Dataset
2. Load Text Data
3. Clean Text Data
4. Develop Vocabulary
5. Save Prepared Data


# Movie Review Dataset

The Movie Review Data is a collection of movie reviews retrieved from the imdb.com website in
the early 2000s by Bo Pang and Lillian Lee. The reviews were collected and made available
as part of their research on natural language processing. The reviews were originally released
in 2002, but an updated and cleaned up version was released in 2004, referred to as v2.0. The
dataset is comprised of 1,000 positive and 1,000 negative movie reviews drawn from an archive
of the rec.arts.movies.reviews newsgroup hosted at IMDB. The authors refer to this dataset as
the polarity dataset.

Our data contains 1000 positive and 1000 negative reviews all written before 2002,
with a cap of 20 reviews per author (312 authors total) per category. We refer to
this corpus as the polarity dataset.
— A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on
Minimum Cuts, 2004.
The data has been cleaned up somewhat, for example:
- The dataset is comprised of only English reviews.
- All text has been converted to lowercase.
- There is white space around punctuation like periods, commas, and brackets.
- Text has been split into one sentence per line.

The data has been used for a few related natural language processing tasks. For classification,
the performance of classical models (such as Support Vector Machines) on the data is in the
range of high 70% to low 80% (e.g. 78%-to-82%). More sophisticated data preparation may see
results as high as 86% with 10-fold cross-validation. This gives us a ballpark of low-to-mid 80s
if we were looking to use this dataset in experiments on modern methods.
... depending on choice of downstream polarity classifier, we can achieve highly
statistically significant improvement (from 82.8% to 86.4%)
— A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based on
Minimum Cuts, 2004.

Dataset can be downloaded from here: 
- Movie Review Polarity Dataset (review polarity.tar.gz, 3MB): http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz

After unzipping the file, you will have a directory called txt sentoken with two subdirectories containing the text neg and pos for negative and positive reviews. Reviews are stored
one per file with a naming convention from cv000 to cv999 for each of neg and pos. Next, let's
look at loading the text data.

# Load Text Data

In this section, we will look at loading individual text files, then processing the directories of
files. We will assume that the review data is downloaded and available in the current working
directory in the folder txt sentoken. We can load an individual text file by opening it, reading
in the ASCII text, and closing the file. This is standard file handling stuff. For example, we can
load the first negative review file cv000 29416.txt as follows:

```
# load one file
filename = 'txt_sentoken/neg/cv000_29416.txt'
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()

```

This loads the document as ASCII and preserves any white space, like new lines. We can
turn this into a function called load doc() that takes a filename of the document to load and
returns the text.

```
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text

```

We have two directories each with 1,000 documents each. We can process each directory in
turn by first getting a list of files in the directory using the listdir() function, then loading
each file in turn. For example, we can load each document in the negative directory using the
load doc() function to do the actual loading.

```
from os import listdir
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# specify directory to load
directory = 'txt_sentoken/neg'
# walk through all files in the folder
for filename in listdir(directory):
# skip files that do not have the right extension
if not filename.endswith(".txt"):
next
# create the full path of the file to open
path = directory + '/' + filename
# load document
doc = load_doc(path)
print('Loaded %s' % filename)

```

Running this example prints the filename of each review after it is loaded.

```
...
Loaded cv995_23113.txt
Loaded cv996_12447.txt
Loaded cv997_5152.txt
Loaded cv998_15691.txt
Loaded cv999_14636.txt
```

We can turn the processing of the documents into a function as well and use it as a template
later for developing a function to clean all documents in a folder. For example, below we define
a process docs() function to do the same thing.

```
from os import listdir
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# load all docs in a directory
def process_docs(directory):
# walk through all files in the folder
for filename in listdir(directory):
# skip files that do not have the right extension
if not filename.endswith(".txt"):
next
# create the full path of the file to open
path = directory + '/' + filename
# load document
doc = load_doc(path)
print('Loaded %s' % filename)
# specify directory to load
directory = 'txt_sentoken/neg'
process_docs(directory)

```

Now that we know how to load the movie review text data, let's look at cleaning it.


### Clean Text Data

In this section, we will look at what data cleaning we might want to do to the movie review
data. We will assume that we will be using a bag-of-words model or perhaps a word embedding
that does not require too much preparation.

### Split into Tokens

First, let's load one document and look at the raw tokens split by white space. We will use the
load doc() function developed in the previous section. We can use the split() function to
split the loaded document into tokens separated by white space.

```
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# load the document
filename = 'txt_sentoken/neg/cv000_29416.txt'
text = load_doc(filename)
# split into tokens by white space
tokens = text.split()
print(tokens)

```

Running the example gives a nice long list of raw tokens from the document.

```
...
'years', 'ago', 'and', 'has', 'been', 'sitting', 'on', 'the', 'shelves', 'ever', 'since',
'.', 'whatever', '.', '.', '.', 'skip', 'it', '!', "where's", 'joblo', 'coming',
'from', '?', 'a', 'nightmare', 'of', 'elm', 'street', '3', '(', '7/10', ')', '-',
'blair', 'witch', '2', '(', '7/10', ')', '-', 'the', 'crow', '(', '9/10', ')', '-',
'the', 'crow', ':', 'salvation', '(', '4/10', ')', '-', 'lost', 'highway', '(',
'10/10', ')', '-', 'memento', '(', '10/10', ')', '-', 'the', 'others', '(', '9/10',
')', '-', 'stir', 'of', 'echoes', '(', '8/10', ')']

```

Just looking at the raw tokens can give us a lot of ideas of things to try, such as:
- Remove punctuation from words (e.g. 'what's').
- Removing tokens that are just punctuation (e.g. '-').
- Removing tokens that contain numbers (e.g. '10/10').
- Remove tokens that have one character (e.g. 'a').
- Remove tokens that don't have much meaning (e.g. 'and').


Some ideas:
- We can filter out punctuation from tokens using regular expressions.
- We can remove tokens that are just punctuation or contain numbers by using an isalpha()
check on each token.
- We can remove English stop words using the list loaded using NLTK.
- We can filter out short tokens by checking their length.

Below is an updated version of cleaning this review.

```
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
# load the document
filename = 'txt_sentoken/neg/cv000_29416.txt'
text = load_doc(filename)
# split into tokens by white space
tokens = text.split()
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
print(tokens)

```

Running the example gives a much cleaner looking list of tokens.

```
...
'explanation', 'craziness', 'came', 'oh', 'way', 'horror', 'teen', 'slasher', 'flick',
'packaged', 'look', 'way', 'someone', 'apparently', 'assuming', 'genre', 'still',
'hot', 'kids', 'also', 'wrapped', 'production', 'two', 'years', 'ago', 'sitting',
'shelves', 'ever', 'since', 'whatever', 'skip', 'wheres', 'joblo', 'coming',
'nightmare', 'elm', 'street', 'blair', 'witch', 'crow', 'crow', 'salvation', 'lost',
'highway', 'memento', 'others', 'stir', 'echoes']

```

We can put this into a function called clean doc() and test it on another review, this time
a positive review.

```
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

Again, the cleaning procedure seems to produce a good set of tokens, at least as a first cut.

```
...
'comic', 'oscar', 'winner', 'martin', 'childs', 'shakespeare', 'love', 'production',
'design', 'turns', 'original', 'prague', 'surroundings', 'one', 'creepy', 'place',
'even', 'acting', 'hell', 'solid', 'dreamy', 'depp', 'turning', 'typically', 'strong',
'performance', 'deftly', 'handling', 'british', 'accent', 'ians', 'holm', 'joe',
'goulds', 'secret', 'richardson', 'dalmatians', 'log', 'great', 'supporting', 'roles',
'big', 'surprise', 'graham', 'cringed', 'first', 'time', 'opened', 'mouth',
'imagining', 'attempt', 'irish', 'accent', 'actually', 'wasnt', 'half', 'bad', 'film',
'however', 'good', 'strong', 'violencegore', 'sexuality', 'language', 'drug', 'content']

```

There are many more cleaning steps we could take and I leave them to your imagination.
Next, let's look at how we can manage a preferred vocabulary of tokens.

Develop Vocabulary

When working with predictive models of text, like a bag-of-words model, there is a pressure to
reduce the size of the vocabulary. The larger the vocabulary, the more sparse the representation
of each word or document. A part of preparing text for sentiment analysis involves defining and
tailoring the vocabulary of words supported by the model. We can do this by loading all of the
documents in the dataset and building a set of words. We may decide to support all of these
words, or perhaps discard some. The final chosen vocabulary can then be saved to file for later
use, such as filtering words in new documents in the future.
We can keep track of the vocabulary in a Counter, which is a dictionary of words and their
count with some additional convenience functions. We need to develop a new function to process
a document and add it to the vocabulary. The function needs to load a document by calling the
previously developed load doc() function. It needs to clean the loaded document using the
previously developed clean doc() function, then it needs to add all the tokens to the Counter,
and update counts. We can do this last step by calling the update() function on the counter
object. Below is a function called add doc to vocab() that takes as arguments a document
filename and a Counter vocabulary.

```
# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
# load doc
doc = load_doc(filename)
# clean doc
tokens = clean_doc(doc)
# update counts
vocab.update(tokens)

```

Finally, we can use our template above for processing all documents in a directory called
process docs() and update it to call add doc to vocab().

```
# load all docs in a directory
def process_docs(directory, vocab):
# walk through all files in the folder
for filename in listdir(directory):
# skip files that do not have the right extension
if not filename.endswith(".txt"):
next
# create the full path of the file to open
path = directory + '/' + filename
# add doc to vocab
add_doc_to_vocab(path, vocab)

```

We can put all of this together and develop a full vocabulary from all documents in the
dataset.

```
import string
import re
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
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
# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
# load doc
doc = load_doc(filename)
# clean doc
tokens = clean_doc(doc)
# update counts
vocab.update(tokens)
# load all docs in a directory
def process_docs(directory, vocab):
# walk through all files in the folder
for filename in listdir(directory):
# skip files that do not have the right extension
if not filename.endswith(".txt"):
next
# create the full path of the file to open
path = directory + '/' + filename
# add doc to vocab
add_doc_to_vocab(path, vocab)
# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))

```

Running the example creates a vocabulary with all documents in the dataset, including
positive and negative reviews. We can see that there are a little over 46,000 unique words across
all reviews and the top 3 words are film, one, and movie.

```
46557
[('film', 8860), ('one', 5521), ('movie', 5440), ('like', 3553), ('even', 2555), ('good',
2320), ('time', 2283), ('story', 2118), ('films', 2102), ('would', 2042), ('much',
2024), ('also', 1965), ('characters', 1947), ('get', 1921), ('character', 1906),
('two', 1825), ('first', 1768), ('see', 1730), ('well', 1694), ('way', 1668), ('make',
1590), ('really', 1563), ('little', 1491), ('life', 1472), ('plot', 1451), ('people',
1420), ('movies', 1416), ('could', 1395), ('bad', 1374), ('scene', 1373), ('never',
1364), ('best', 1301), ('new', 1277), ('many', 1268), ('doesnt', 1267), ('man', 1266),
('scenes', 1265), ('dont', 1210), ('know', 1207), ('hes', 1150), ('great', 1141),
('another', 1111), ('love', 1089), ('action', 1078), ('go', 1075), ('us', 1065),
('director', 1056), ('something', 1048), ('end', 1047), ('still', 1038)]

```

Perhaps the least common words, those that only appear once across all reviews, are not
predictive. Perhaps some of the most common words are not useful too. These are good
questions and really should be tested with a specific predictive model. Generally, words that
only appear once or a few times across 2,000 reviews are probably not predictive and can be
removed from the vocabulary, greatly cutting down on the tokens we need to model. We can do
this by stepping through words and their counts and only keeping those with a count above a
chosen threshold. Here we will use 5 occurrences.

```
# keep tokens with > 5 occurrence
min_occurrence = 5
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
print(len(tokens))

```

This reduces the vocabulary from 46,557 to 14,803 words, a huge drop. Perhaps a minimum
of 5 occurrences is too aggressive; you can experiment with different values. We can then save
the chosen vocabulary of words to a new file. I like to save the vocabulary as ASCII with one
word per line. Below defines a function called save list() to save a list of items, in this case,
tokens to file, one per line.

```
def save_list(lines, filename):
data = '\n'.join(lines)
file = open(filename, 'w')
file.write(data)
file.close()

```

The complete example for defining and saving the vocabulary is listed below.

```
import string
import re
from os import listdir
from collections import Counter
from nltk.corpus import stopwords
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
# load doc and add to vocab
def add_doc_to_vocab(filename, vocab):
# load doc
doc = load_doc(filename)
# clean doc
tokens = clean_doc(doc)
# update counts
vocab.update(tokens)
# load all docs in a directory
def process_docs(directory, vocab):
# walk through all files in the folder
for filename in listdir(directory):
# skip files that do not have the right extension
if not filename.endswith(".txt"):
next
# create the full path of the file to open
path = directory + '/' + filename
# add doc to vocab
add_doc_to_vocab(path, vocab)
# save list to file
def save_list(lines, filename):
data = '\n'.join(lines)
file = open(filename, 'w')
file.write(data)
file.close()
# define vocab
vocab = Counter()
# add all docs to vocab
process_docs('txt_sentoken/neg', vocab)
process_docs('txt_sentoken/pos', vocab)
# print the size of the vocab
print(len(vocab))
# print the top words in the vocab
print(vocab.most_common(50))
# keep tokens with > 5 occurrence
min_occurrence = 5
tokens = [k for k,c in vocab.items() if c >= min_occurrence]
print(len(tokens))
# save tokens to a vocabulary file
save_list(tokens, 'vocab.txt')

```

Running this final snippet after creating the vocabulary will save the chosen words to file. It
is a good idea to take a look at, and even study, your chosen vocabulary in order to get ideas
for better preparing this data, or text data in the future.

```
hasnt
updating
figuratively
symphony
civilians
might
fisherman
hokum
witch
buffoons
...

```

Next, we can look at using the vocabulary to create a prepared version of the movie review
dataset.

### Save Prepared Data

We can use the data cleaning and chosen vocabulary to prepare each movie review and save the
prepared versions of the reviews ready for modeling. This is a good practice as it decouples
the data preparation from modeling, allowing you to focus on modeling and circle back to data
prep if you have new ideas. We can start off by loading the vocabulary from vocab.txt.

```
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# load vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
```

Next, we can clean the reviews, use the loaded vocab to filter out unwanted tokens, and
save the clean reviews in a new file. One approach could be to save all the positive reviews
in one file and all the negative reviews in another file, with the filtered tokens separated by
white space for each review on separate lines. First, we can define a function to process a
document, clean it, filter it, and return it as a single line that could be saved in a file. Below
defines the doc to line() function to do just that, taking a filename and vocabulary (as a set)
as arguments. It calls the previously defined load doc() function to load the document and
clean doc() to tokenize the document.

```
# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
# load the doc
doc = load_doc(filename)
# clean doc
tokens = clean_doc(doc)
# filter by vocab
tokens = [w for w in tokens if w in vocab]
return ' '.join(tokens)

```

Next, we can define a new version of process docs() to step through all reviews in a folder
and convert them to lines by calling doc to line() for each document. A list of lines is then
returned.

```
# load all docs in a directory
def process_docs(directory, vocab):
lines = list()
# walk through all files in the folder
for filename in listdir(directory):
# skip files that do not have the right extension
if not filename.endswith(".txt"):
next
# create the full path of the file to open
path = directory + '/' + filename
# load and clean the doc
line = doc_to_line(path, vocab)
# add to list
lines.append(line)
return lines

```

We can then call process docs() for both the directories of positive and negative reviews,
then call save list() from the previous section to save each list of processed reviews to a file.

# Save Prepared Data
The complete code listing is provided below.

```
import string
import re
from os import listdir
from nltk.corpus import stopwords
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
# save list to file
def save_list(lines, filename):
data = '\n'.join(lines)
file = open(filename, 'w')
file.write(data)
file.close()
# load doc, clean and return line of tokens
def doc_to_line(filename, vocab):
# load the doc
doc = load_doc(filename)
# clean doc
tokens = clean_doc(doc)
# filter by vocab
tokens = [w for w in tokens if w in vocab]
return ' '.join(tokens)
# load all docs in a directory
def process_docs(directory, vocab):
lines = list()
# walk through all files in the folder
for filename in listdir(directory):
# skip files that do not have the right extension
if not filename.endswith(".txt"):
next
# create the full path of the file to open
path = directory + '/' + filename
# load and clean the doc
line = doc_to_line(path, vocab)
# add to list
lines.append(line)
return lines
# load vocabulary
vocab_filename = 'vocab.txt'
vocab = load_doc(vocab_filename)
vocab = vocab.split()
vocab = set(vocab)
# prepare negative reviews
negative_lines = process_docs('txt_sentoken/neg', vocab)
save_list(negative_lines, 'negative.txt')
# prepare positive reviews
positive_lines = process_docs('txt_sentoken/pos', vocab)
save_list(positive_lines, 'positive.txt')

```

Running the example saves two new files, negative.txt and positive.txt, that contain the
prepared negative and positive reviews respectively. The data is ready for use in a bag-of-words
or even word embedding model.



##### Run Notebook
Click notebook `01_load_file.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `02_load_all_files.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `03_load_all_files_with_func.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `04_load_and_split.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `05_clean_review.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `06_clean_review_func.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `07_clean_and_build_vocab.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `08_build_vocabulary.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `09_load_vocab.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `10_save_clean_filtered_reviews.ipynb` in jupterLab UI and run jupyter notebook.

# Further Reading

This section provides more resources on the topic if you are looking go deeper.

Dataset

- Movie Review Data.
http://www.cs.cornell.edu/people/pabo/movie-review-data/
- A Sentimental Education: Sentiment Analysis Using Subjectivity Summarization Based
on Minimum Cuts, 2004.
http://xxx.lanl.gov/abs/cs/0409058
- Movie Review Polarity Dataset.
http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.
gz
- Dataset Readme v2.0 and v1.1.
http://www.cs.cornell.edu/people/pabo/movie-review-data/poldata.README.2.0.
txt
http://www.cs.cornell.edu/people/pabo/movie-review-data/README.1.1

APIs

- nltk.tokenize package API.
http://www.nltk.org/api/nltk.tokenize.html
- Chapter 2, Accessing Text Corpora and Lexical Resources.
http://www.nltk.org/book/ch02.html
- os API Miscellaneous operating system interfaces.
https://docs.python.org/3/library/os.html
- collections API - Container datatypes.
https://docs.python.org/3/library/collections.html

# Summary

In this tutorial, you discovered how to prepare movie review text data for sentiment analysis,
step-by-step. Specifically, you learned:
- How to load text data and clean it to remove punctuation and other non-words.
- How to develop a vocabulary, tailor it, and save it to file.
- How to prepare movie reviews using cleaning and a predefined vocabulary and save them
to new files ready for modeling.

# Next

In the next chapter, you will discover how you can develop a neural bag-of-words model for
movie review sentiment analysis.
