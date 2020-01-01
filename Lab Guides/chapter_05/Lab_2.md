# How to Clean Text Manually and with NLTK
You cannot go straight from raw text to fitting a machine learning or deep learning model. You
must clean your text first, which means splitting it into words and handling punctuation and
case. In fact, there is a whole suite of text preparation methods that you may need to use, and
the choice of methods really depends on your natural language processing task. In this tutorial,
you will discover how you can clean and prepare your text ready for modeling with machine
learning. After completing this tutorial, you will know:
- How to get started by developing your own very simple text cleaning tools.
- How to take a step up and use the more sophisticated methods in the NLTK library.
- Considerations when preparing text for natural language processing models.

Let's get started.

# Tutorial Overview

This tutorial is divided into the following parts:
1. Metamorphosis by Franz Kafka
2. Text Cleaning is Task Specific
3. Manual Tokenization
4. Tokenization and Cleaning with NLTK
5. Additional Text Cleaning Considerations

35

5.2. Metamorphosis by Franz Kafka

5.2

36

Metamorphosis by Franz Kafka

Let's start off by selecting a dataset. In this tutorial, we will use the text from the book
Metamorphosis by Franz Kafka. No specific reason, other than it's short, I like it, and you may
like it too. I expect it's one of those classics that most students have to read in school. The full
text for Metamorphosis is available for free from Project Gutenberg. You can download the
ASCII text version of the text here:
- Metamorphosis by Franz Kafka Plain Text UTF-8 (may need to load the page twice).
http://www.gutenberg.org/cache/epub/5200/pg5200.txt

Download the file and place it in your current working directory with the file name
metamorphosis.txt. The file contains header and footer information that we are not interested in, specifically copyright and license information. Open the file and delete the header
and footer information and save the file as metamorphosis clean.txt. The start of the clean
file should look like:
One morning, when Gregor Samsa woke from troubled dreams, he found himself
transformed in his bed into a horrible vermin.
The file should end with:
And, as if in confirmation of their new dreams and good intentions, as soon as they
reached their destination Grete was the first to get up and stretch out her young
body.
Poor Gregor...

5.3

Text Cleaning Is Task Specific

After actually getting a hold of your text data, the first step in cleaning up text data is to have
a strong idea about what you're trying to achieve, and in that context review your text to see
what exactly might help. Take a moment to look at the text. What do you notice? Here's what
I see:
- It's plain text so there is no markup to parse (yay!).
- The translation of the original German uses UK English (e.g. travelling).
- The lines are artificially wrapped with new lines at about 70 characters (meh).
- There are no obvious typos or spelling mistakes.
- There's punctuation like commas, apostrophes, quotes, question marks, and more.
- There's hyphenated descriptions like armour-like.
- There's a lot of use of the em dash (-) to continue sentences (maybe replace with commas?).
- There are names (e.g. Mr. Samsa)

5.4. Manual Tokenization

37

- There does not appear to be numbers that require handling (e.g. 1999)
- There are section markers (e.g. II and III ).

I'm sure there is a lot more going on to the trained eye. We are going to look at general
text cleaning steps in this tutorial. Nevertheless, consider some possible objectives we may have
when working with this text document. For example:
- If we were interested in developing a Kafkaesque language model, we may want to keep all
of the case, quotes, and other punctuation in place.
- If we were interested in classifying documents as Kafka and Not Kafka, maybe we would
want to strip case, punctuation, and even trim words back to their stem.

Use your task as the lens by which to choose how to ready your text data.

5.4

Manual Tokenization

Text cleaning is hard, but the text we have chosen to work with is pretty clean already. We
could just write some Python code to clean it up manually, and this is a good exercise for those
simple problems that you encounter. Tools like regular expressions and splitting strings can get
you a long way.

5.4.1

Load Data

Let's load the text data so that we can work with it. The text is small and will load quickly
and easily fit into memory. This will not always be the case and you may need to write code
to memory map the file. Tools like NLTK (covered in the next section) will make working
with large files much easier. We can load the entire metamorphosis clean.txt into memory as
follows:

```
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
```

5.4.2

Split by Whitespace

Clean text often means a list of words or tokens that we can work with in our machine learning
models. This means converting the raw text into a list of words and saving it again. A very
simple way to do this would be to split the document by white space, including “ ” (space), new
lines, tabs and more. We can do this in Python with the split() function on the loaded string.

```
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
print(words[:100])
```

Running the example splits the document into a long list of words and prints the first 100 for
us to review. We can see that punctuation is preserved (e.g. wasn't and armour-like), which is
nice. We can also see that end of sentence punctuation is kept with the last word (e.g. thought.),
which is not great.

```
['One', 'morning,', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams,', 'he',
'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible',
'vermin.', 'He', 'lay', 'on', 'his', 'armour-like', 'back,', 'and', 'if', 'he',
'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly,',
'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections.',
'The', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed',
'ready', 'to', 'slide', 'off', 'any', 'moment.', 'His', 'many', 'legs,', 'pitifully',
'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him,', 'waved',
'about', 'helplessly', 'as', 'he', 'looked.', '"What\'s', 'happened', 'to', 'me?"',
'he', 'thought.', 'It', "wasn't", 'a', 'dream.', 'His', 'room,', 'a', 'proper', 'human']
```


Select Words

Another approach might be to use the regex model (re) and split the document into words by
selecting for strings of alphanumeric characters (a-z, A-Z, 0-9 and ' '). For example:
import re

```
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split based on words only
words = re.split(r'\W+', text)
print(words[:100])
```

Again, running the example we can see that we get our list of words. This time, we can see
that armour-like is now two words armour and like (fine) but contractions like What's is also
two words What and s (not great).
['One', 'morning', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams', 'he',
'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible',
'vermin', 'He', 'lay', 'on', 'his', 'armour', 'like', 'back', 'and', 'if', 'he',
'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly',
'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections',
'The', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed',
'ready', 'to', 'slide', 'off', 'any', 'moment', 'His', 'many', 'legs', 'pitifully',
'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him', 'waved',
'about', 'helplessly', 'as', 'he', 'looked', 'What', 's', 'happened', 'to', 'me', 'he',
'thought', 'It', 'wasn', 't', 'a', 'dream', 'His', 'room']

5.4. Manual Tokenization

39
```


5.4.4

Split by Whitespace and Remove Punctuation

We may want the words, but without the punctuation like commas and quotes. We also want to
keep contractions together. One way would be to split the document into words by white space
(as in the section Split by Whitespace), then use string translation to replace all punctuation with
nothing (e.g. remove it). Python provides a constant called string.punctuation that provides a
great list of punctuation characters. For example:

```
print(string.punctuation)
```

Results in:

```
!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
```

We can use regular expressions to select for the punctuation characters and use the sub()
function to replace them with nothing. For example:
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# remove punctuation from each word
stripped = [re_punc.sub('', w) for w in words]
```

We can put all of this together, load the text file, split it into words by white space, then
translate each word to remove the punctuation.

```
import string
import re
# load text
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# remove punctuation from each word
stripped = [re_punc.sub('', w) for w in words]
print(stripped[:100])
```

We can see that this has had the desired effect, mostly. Contractions like What's have
become Whats but armour-like has become armourlike.

5.4. Manual Tokenization

```
['One', 'morning', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams', 'he',
'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible',
'vermin', 'He', 'lay', 'on', 'his', 'armourlike', 'back', 'and', 'if', 'he', 'lifted',
'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly',
'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections',
'The', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed',
'ready', 'to', 'slide', 'off', 'any', 'moment', 'His', 'many', 'legs', 'pitifully',
'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him', 'waved',
'about', 'helplessly', 'as', 'he', 'looked', 'Whats', 'happened', 'to', 'me', 'he',
'thought', 'It', 'wasnt', 'a', 'dream', 'His', 'room', 'a', 'proper', 'human']
```

Sometimes text data may contain non-printable characters. We can use a similar approach to
filter out all non-printable characters by selecting the inverse of the string.printable constant.
For example:

```
...
re_print = re.compile('[^%s]' % re.escape(string.printable))
result = [re_print.sub('', w) for w in words]
```

5.4.5

Normalizing Case

It is common to convert all words to one case. This means that the vocabulary will shrink in
size, but some distinctions are lost (e.g. Apple the company vs apple the fruit is a commonly
used example). We can convert all words to lowercase by calling the lower() function on each
word. For example:

```
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words by white space
words = text.split()
# convert to lower case
words = [word.lower() for word in words]
print(words[:100])
```

Running the example, we can see that all words are now lowercase.

```
['one', 'morning,', 'when', 'gregor', 'samsa', 'woke', 'from', 'troubled', 'dreams,', 'he',
'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible',
'vermin.', 'he', 'lay', 'on', 'his', 'armour-like', 'back,', 'and', 'if', 'he',
'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly,',
'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections.',
'the', 'bedding', 'was', 'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed',
'ready', 'to', 'slide', 'off', 'any', 'moment.', 'his', 'many', 'legs,', 'pitifully',
'thin', 'compared', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him,', 'waved',
'about', 'helplessly', 'as', 'he', 'looked.', '"what\'s', 'happened', 'to', 'me?"',
'he', 'thought.', 'it', "wasn't", 'a', 'dream.', 'his', 'room,', 'a', 'proper', 'human']
```


5.5. Tokenization and Cleaning with NLTK

Note on Cleaning Text

Cleaning text is really hard, problem specific, and full of tradeoffs. Remember, simple is better.
Simpler text data, simpler models, smaller vocabularies. You can always make things more
complex later to see if it results in better model skill. Next, we'll look at some of the tools in
the NLTK library that offer more than simple string splitting.

Tokenization and Cleaning with NLTK

The Natural Language Toolkit, or NLTK for short, is a Python library written for working and
modeling text. It provides good tools for loading and cleaning text that we can use to get our
data ready for working with machine learning and deep learning algorithms.

5.5.1

Install NLTK

You can install NLTK using your favorite package manager, such as pip. On a POSIX-compatible
machine, this would be:

```
sudo pip install -U nltk
```

After installation, you will need to install the data used with the library, including a great
set of documents that you can use later for testing other tools in NLTK. There are few ways to
do this, such as from within a script:

```
import nltk
nltk.download()
```

Or from the command line:

```
python -m nltk.downloader all
```


5.5.2

Split into Sentences

A good useful first step is to split the text into sentences. Some modeling tasks prefer input
to be in the form of paragraphs or sentences, such as Word2Vec. You could first split your
text into sentences, split each sentence into words, then save each sentence to file, one per line.
NLTK provides the sent tokenize() function to split text into sentences. The example below
loads the metamorphosis clean.txt file into memory, splits it into sentences, and prints the
first sentence.

```
from nltk import sent_tokenize
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into sentences
sentences = sent_tokenize(text)
print(sentences[0])
```

Running the example, we can see that although the document is split into sentences, that
each sentence still preserves the new line from the artificial wrap of the lines in the original
document.

```
One morning, when Gregor Samsa woke from troubled dreams, he found
himself transformed in his bed into a horrible vermin.
```


5.5.3

Split into Words

NLTK provides a function called word tokenize() for splitting strings into tokens (nominally
words). It splits tokens based on white space and punctuation. For example, commas and
periods are taken as separate tokens. Contractions are split apart (e.g. What's becomes What
and 's). Quotes are kept, and so on. For example:

```
from nltk.tokenize import word_tokenize
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
tokens = word_tokenize(text)
print(tokens[:100])
```

Running the code, we can see that punctuation are now tokens that we could then decide to
specifically filter out.

```
['One', 'morning', ',', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams',
',', 'he', 'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a',
'horrible', 'vermin', '.', 'He', 'lay', 'on', 'his', 'armour-like', 'back', ',', 'and',
'if', 'he', 'lifted', 'his', 'head', 'a', 'little', 'he', 'could', 'see', 'his',
'brown', 'belly', ',', 'slightly', 'domed', 'and', 'divided', 'by', 'arches', 'into',
'stiff', 'sections', '.', 'The', 'bedding', 'was', 'hardly', 'able', 'to', 'cover',
'it', 'and', 'seemed', 'ready', 'to', 'slide', 'off', 'any', 'moment', '.', 'His',
'many', 'legs', ',', 'pitifully', 'thin', 'compared', 'with', 'the', 'size', 'of',
'the', 'rest', 'of', 'him', ',', 'waved', 'about', 'helplessly', 'as', 'he', 'looked',
'.', '``', 'What', "'s", 'happened', 'to']
```


5.5.4

Filter Out Punctuation

We can filter out all tokens that we are not interested in, such as all standalone punctuation. This
can be done by iterating over all tokens and only keeping those tokens that are all alphabetic.
Python has the function isalpha() that can be used. For example:

5.5. Tokenization and Cleaning with NLTK

```
from nltk.tokenize import word_tokenize
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
tokens = word_tokenize(text)
# remove all tokens that are not alphabetic
words = [word for word in tokens if word.isalpha()]
print(words[:100])
```

Running the example, you can see that not only punctuation tokens, but examples like
armour-like and 's were also filtered out.

```
['One', 'morning', 'when', 'Gregor', 'Samsa', 'woke', 'from', 'troubled', 'dreams', 'he',
'found', 'himself', 'transformed', 'in', 'his', 'bed', 'into', 'a', 'horrible',
'vermin', 'He', 'lay', 'on', 'his', 'back', 'and', 'if', 'he', 'lifted', 'his', 'head',
'a', 'little', 'he', 'could', 'see', 'his', 'brown', 'belly', 'slightly', 'domed',
'and', 'divided', 'by', 'arches', 'into', 'stiff', 'sections', 'The', 'bedding', 'was',
'hardly', 'able', 'to', 'cover', 'it', 'and', 'seemed', 'ready', 'to', 'slide', 'off',
'any', 'moment', 'His', 'many', 'legs', 'pitifully', 'thin', 'compared', 'with', 'the',
'size', 'of', 'the', 'rest', 'of', 'him', 'waved', 'about', 'helplessly', 'as', 'he',
'looked', 'What', 'happened', 'to', 'me', 'he', 'thought', 'It', 'was', 'a', 'dream',
'His', 'room', 'a', 'proper', 'human', 'room']
```


5.5.5

Filter out Stop Words (and Pipeline)

Stop words are those words that do not contribute to the deeper meaning of the phrase. They
are the most common words such as: the, a, and is. For some applications like documentation
classification, it may make sense to remove stop words. NLTK provides a list of commonly
agreed upon stop words for a variety of languages, such as English. They can be loaded as
follows:

```
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
print(stop_words)
```

You can see the full list as follows:

```
['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers',
'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are',
'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does',
'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until',
'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down',
'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here',
'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more',
'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd',
'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn',
'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn',
'weren', 'won', 'wouldn']
```

You can see that they are all lower case and have punctuation removed. You could compare
your tokens to the stop words and filter them out, but you must ensure that your text is prepared
the same way. Let's demonstrate this with a small pipeline of text preparation including:
- Load the raw text.
- Split into tokens.
- Convert to lowercase.
- Remove punctuation from each token.
- Filter out remaining tokens that are not alphabetic.
- Filter out tokens that are stop words.

```
import string
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
tokens = word_tokenize(text)
# convert to lower case
tokens = [w.lower() for w in tokens]
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
# remove punctuation from each word
stripped = [re_punc.sub('', w) for w in tokens]
# remove remaining tokens that are not alphabetic
words = [word for word in stripped if word.isalpha()]
# filter out stop words
stop_words = set(stopwords.words('english'))
words = [w for w in words if not w in stop_words]
print(words[:100])
```

Running this example, we can see that in addition to all of the other transforms, stop words
like a and to have been removed. I note that we are still left with tokens like nt. The rabbit
hole is deep; there's always more we can do.


```
['one', 'morning', 'gregor', 'samsa', 'woke', 'troubled', 'dreams', 'found', 'transformed',
'bed', 'horrible', 'vermin', 'lay', 'armourlike', 'back', 'lifted', 'head', 'little',
'could', 'see', 'brown', 'belly', 'slightly', 'domed', 'divided', 'arches', 'stiff',
'sections', 'bedding', 'hardly', 'able', 'cover', 'seemed', 'ready', 'slide', 'moment',
'many', 'legs', 'pitifully', 'thin', 'compared', 'size', 'rest', 'waved', 'helplessly',
'looked', 'happened', 'thought', 'nt', 'dream', 'room', 'proper', 'human', 'room',
'although', 'little', 'small', 'lay', 'peacefully', 'four', 'familiar', 'walls',
'collection', 'textile', 'samples', 'lay', 'spread', 'table', 'samsa', 'travelling',
'salesman', 'hung', 'picture', 'recently', 'cut', 'illustrated', 'magazine', 'housed',
'nice', 'gilded', 'frame', 'showed', 'lady', 'fitted', 'fur', 'hat', 'fur', 'boa',
'sat', 'upright', 'raising', 'heavy', 'fur', 'muff', 'covered', 'whole', 'lower',
'arm', 'towards', 'viewer']
```


Stem Words

Stemming refers to the process of reducing each word to its root or base. For example fishing,
fished, fisher all reduce to the stem fish. Some applications, like document classification, may
benefit from stemming in order to both reduce the vocabulary and to focus on the sense or
sentiment of a document rather than deeper meaning. There are many stemming algorithms,
although a popular and long-standing method is the Porter Stemming algorithm. This method
is available in NLTK via the PorterStemmer class. For example:
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

```
# load data
filename = 'metamorphosis_clean.txt'
file = open(filename, 'rt')
text = file.read()
file.close()
# split into words
tokens = word_tokenize(text)
# stemming of words
porter = PorterStemmer()
stemmed = [porter.stem(word) for word in tokens]
print(stemmed[:100])
```

Running the example, you can see that words have been reduced to their stems, such as
trouble has become troubl. You can also see that the stemming implementation has also reduced
the tokens to lowercase, likely for internal look-ups in word tables.

```
['one', 'morn', ',', 'when', 'gregor', 'samsa', 'woke', 'from', 'troubl', 'dream', ',',
'he', 'found', 'himself', 'transform', 'in', 'hi', 'bed', 'into', 'a', 'horribl',
'vermin', '.', 'He', 'lay', 'on', 'hi', 'armour-lik', 'back', ',', 'and', 'if', 'he',
'lift', 'hi', 'head', 'a', 'littl', 'he', 'could', 'see', 'hi', 'brown', 'belli', ',',
'slightli', 'dome', 'and', 'divid', 'by', 'arch', 'into', 'stiff', 'section', '.',
'the', 'bed', 'wa', 'hardli', 'abl', 'to', 'cover', 'it', 'and', 'seem', 'readi', 'to',
'slide', 'off', 'ani', 'moment', '.', 'hi', 'mani', 'leg', ',', 'piti', 'thin',
'compar', 'with', 'the', 'size', 'of', 'the', 'rest', 'of', 'him', ',', 'wave',
'about', 'helplessli', 'as', 'he', 'look', '.', '``', 'what', "'s", 'happen', 'to'
```

There is a nice suite of stemming and lemmatization algorithms to choose from in NLTK, if
reducing words to their root is something you need for your project.

5.6

Additional Text Cleaning Considerations

We are only getting started. Because the source text for this tutorial was reasonably clean to
begin with, we skipped many concerns of text cleaning that you may need to deal with in your
own project. Here is a shortlist of additional considerations when cleaning text:
- Handling large documents and large collections of text documents that do not fit into
memory.
- Extracting text from markup like HTML, PDF, or other structured document formats.
- Transliteration of characters from other languages into English.
- Decoding Unicode characters into a normalized form, such as UTF8.
- Handling of domain specific words, phrases, and acronyms.
- Handling or removing numbers, such as dates and amounts.
- Locating and correcting common typos and misspellings.
- And much more...

The list could go on. Hopefully, you can see that getting truly clean text is impossible, that
we are really doing the best we can based on the time, resources, and knowledge we have. The
idea of clean is really defined by the specific task or concern of your project.
A pro tip is to continually review your tokens after every transform. I have tried to show
that in this tutorial and I hope you take that to heart. Ideally, you would save a new file after
each transform so that you can spend time with all of the data in the new form. Things always
jump out at you when to take the time to review your data.

# Further Reading

This section provides more resources on the topic if you are looking go deeper.
- Metamorphosis by Franz Kafka on Project Gutenberg.
http://www.gutenberg.org/ebooks/5200
- Installing NLTK.
http://www.nltk.org/install.html
- Installing NLTK Data.
http://www.nltk.org/data.html
- Python isalpha() function.
https://docs.python.org/3/library/stdtypes.html#str.isalpha
- Stop Words on Wikipedia.
https://en.wikipedia.org/wiki/Stop_words
- Stemming on Wikipedia.
https://en.wikipedia.org/wiki/Stemming
- nltk.tokenize package API.
http://www.nltk.org/api/nltk.tokenize.html
- Porter Stemming algorithm.
https://tartarus.org/martin/PorterStemmer/
- nltk.stem package API.
http://www.nltk.org/api/nltk.stem.html
- Processing Raw Text, Natural Language Processing with Python.
http://www.nltk.org/book/ch03.html

# Summary

In this tutorial, you discovered how to clean text or machine learning in Python.
Specifically, you learned:
- How to get started by developing your own very simple text cleaning tools.
- How to take a step up and use the more sophisticated methods in the NLTK library.
- Considerations when preparing text for natural language processing models.

# Next
In the next chapter, you will discover how you can encode text data using the scikit-learn
Python library.

##### Run Notebook
Click notebook `01_manual_load_data.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `02_manual_split.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `03_manual_select_words.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `04_manual_remove_punctuation.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `05_manual_normalize_case.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `06_nltk_download.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `07_nltk_split_sentences.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `08_nltk_split_words.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `09_nltk_remove_punctuation.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `10_nltk_stop_words.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `11_nltk_filter_stop_words.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `12_nltk_stemming.ipynb` in jupterLab UI and run jupyter notebook.

