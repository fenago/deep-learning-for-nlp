<img align="right" src="../logo-small.png">



# How to Prepare a Photo Caption Dataset For Modeling
Automatic photo captioning is a problem where a model must generate a human-readable textual
description given a photograph. It is a challenging problem in artificial intelligence that requires
both image understanding from the field of computer vision as well as language generation from
the field of natural language processing. It is now possible to develop your own image caption
models using deep learning and freely available datasets of photos and their descriptions. In this
tutorial, you will discover how to prepare photos and textual descriptions ready for developing
a deep learning automatic photo caption generation model. After completing this tutorial, you
will know:
- About the Flickr8K dataset comprised of more than 8,000 photos and up to 5 captions for
each photo.
- How to generally load and prepare photo and text data for modeling with deep learning.
- How to specifically encode data for two different types of deep learning models in Keras.

Let's get started.

#### Pre-reqs:
- Google Chrome (Recommended)

#### Lab Environment
Notebooks are ready to run. All packages have been installed. There is no requirement for any setup. Please download datasets specified in instructions to run notebooks in this lab.

**Note:** Elev8ed Notebooks (powered by Jupyter) will be accessible at the port given to you by your instructor. Password for jupyterLab : `1234`

All Notebooks are present in `work/deep-learning-for-nlp` folder.

You can access jupyter lab at `<host-ip>:<port>/lab/workspaces/lab16_Photo_Caption_Dataset`


#### Tutorial Overview

This tutorial is divided into the following parts:
1. Download the Flickr8K Dataset
2. How to Load Photographs
3. Pre-Calculate Photo Features
4. How to Load Descriptions
5. Prepare Description Text
6. Whole Description Sequence Model
7. Word-By-Word Model
8. Progressive Loading

###### Download the Flickr8K Dataset

A good dataset to use when getting started with image captioning is the Flickr8K dataset. The
reason is that it is realistic and relatively small so that you can download it and build models on
your workstation using a CPU. The definitive description of the dataset is in the paper Framing
Image Description as a Ranking Task: Data, Models and Evaluation Metrics from 2013. The
authors describe the dataset as follows:
We introduce a new benchmark collection for sentence-based image description and
search, consisting of 8,000 images that are each paired with five different captions
which provide clear descriptions of the salient entities and events.

...
The images were chosen from six different Flickr groups, and tend not to contain
any well-known people or locations, but were manually selected to depict a variety
of scenes and situations.
— Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics, 2013.
The dataset is available for free. 

#### Download Dataset
Dataset is very huge. Before running the notebook, **download** the dataset and unzip it.

`curl -L  https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -o Flickr8k_Dataset.zip`

`unzip Flickr8k_Dataset.zip`
 
Within a short time, you will receive an email that contains links to two files:
- Flickr8k Dataset.zip (1 Gigabyte) An archive of all photographs.
- Flickr8k text.zip (2.2 Megabytes) An archive of all text descriptions for photographs.

Download the datasets and unzip them into your current working directory. You will have
two directories:
- Flicker8k Dataset: Contains more than 8000 photographs in JPEG format (yes the
directory name spells it 'Flicker' not 'Flickr').
- Flickr8k text: Contains a number of files containing different sources of descriptions for
the photographs.

Next, let's look at how to load the images.

# How to Load Photographs

In this section, we will develop some code to load the photos for use with the Keras deep learning
library in Python. The image file names are unique image identifiers. For example, here is a
sample of image file names:

```
990890291_afc72be141.jpg
99171998_7cc800ceef.jpg
99679241_adc853a5c0.jpg
997338199_7343367d7f.jpg
997722733_0cb5439472.jpg

```

Keras provides the load img() function that can be used to load the image files directly as
an array of pixels.

```
from keras.preprocessing.image import load_img
image = load_img('990890291_afc72be141.jpg')

```

The pixel data needs to be converted to a NumPy array for use in Keras. We can use the
img to array() Keras function to convert the loaded data.
from keras.preprocessing.image import img_to_array

```
image = img_to_array(image)

```

We may want to use a pre-defined feature extraction model, such as a state-of-the-art deep
image classification network trained on Image net. The Oxford Visual Geometry Group (VGG)
model is popular for this purpose and is available in Keras. If we decide to use this pre-trained
model as a feature extractor in our model, we can pre-process the pixel data for the model by
using the preprocess input() function in Keras, for example:

```
from keras.applications.vgg16 import preprocess_input
# reshape data into a single sample of an image
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)

```

We may also want to force the loading of the photo to have the same pixel dimensions as the
VGG model, which are 224 x 224 pixels. We can do that in the call to load img(), for example:

```
image = load_img('990890291_afc72be141.jpg', target_size=(224, 224))

```

We may want to extract the unique image identifier from the image filename. We can do
that by splitting the filename string by the '.' (period) character and retrieving the first element
of the resulting array:

```
image_id = filename.split('.')[0]

```


# Pre-Calculate Photo Features

We can tie all of this together and develop a function that, given the name of the directory
containing the photos, will load and pre-process all of the photos for the VGG model and return
them in a dictionary keyed on their unique image identifiers.

```

from os import listdir
from os import path
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

def load_photos(directory):
images = dict()
for name in listdir(directory):
# load an image from file
filename = path.join(directory, name)
image = load_img(filename, target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# get image id
image_id = name.split('.')[0]
images[image_id] = image
return images
# load images
directory = 'Flicker8k_Dataset'
images = load_photos(directory)
print('Loaded Images: %d' % len(images))

```

Running this example prints the number of loaded images. It takes a few minutes to run.

```
Loaded Images: 8091
```


25.4

Pre-Calculate Photo Features

It is possible to use a pre-trained model to extract the features from photos in the dataset and
store the features to file. This is an efficiency that means that the language part of the model
that turns features extracted from the photo into textual descriptions can be trained standalone
from the feature extraction model. The benefit is that the very large pre-trained models do not
need to be loaded, held in memory, and used to process each photo while training the language
model.
Later, the feature extraction model and language model can be put back together for making
predictions on new photos. In this section, we will extend the photo loading behavior developed
in the previous section to load all photos, extract their features using a pre-trained VGG model,
and store the extracted features to a new file that can be loaded and used to train the language
model. The first step is to load the VGG model. This model is provided directly in Keras and
can be loaded as follows. Note that this will download the 500-megabyte model weights to your
computer, which may take a few minutes.

```
from keras.applications.vgg16 import VGG16
# load the model
in_layer = Input(shape=(224, 224, 3))
model = VGG16(include_top=False, input_tensor=in_layer, pooling='avg')
model.summary()

```

This will load the VGG 16-layer model. The two Dense output layers as well as the
classification output layer are removed from the model by setting include top=False. The
output from the final pooling layer is taken as the features extracted from the image. Next, we
can walk over all images in the directory of images as in the previous section and call predict()
function on the model for each prepared image to get the extracted features. The features can
then be stored in a dictionary keyed on the image id. The complete example is listed below.

```
from os import listdir
from os import path
from pickle import dump
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.layers import Input

# extract features from each photo in the directory
def extract_features(directory):
# load the model
in_layer = Input(shape=(224, 224, 3))
model = VGG16(include_top=False, input_tensor=in_layer)
model.summary()
# extract features from each photo
features = dict()
for name in listdir(directory):
# load an image from file
filename = path.join(directory, name)
image = load_img(filename, target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# get features
feature = model.predict(image, verbose=0)
# get image id
image_id = name.split('.')[0]
# store feature
features[image_id] = feature
print('>%s' % name)
return features
# extract features from all images
directory = 'Flicker8k_Dataset'
features = extract_features(directory)
print('Extracted Features: %d' % len(features))
# save to file
dump(features, open('features.pkl', 'wb'))

```

The example may take some time to complete, perhaps one hour. After all features are
extracted, the dictionary is stored in the file features.pkl in the current working directory.
These features can then be loaded later and used as input for training a language model. You
could experiment with other types of pre-trained models in Keras.

25.5

How to Load Descriptions

It is important to take a moment to talk about the descriptions; there are a number available.
The file Flickr8k.token.txt contains a list of image identifiers (used in the image filenames)
and tokenized descriptions. Each image has multiple descriptions. Below is a sample of the
descriptions from the file showing 5 different descriptions for a single image.

```
1305564994_00513f9a5b.jpg#0
racer 's motorbike .
1305564994_00513f9a5b.jpg#1
1305564994_00513f9a5b.jpg#2
design and color .
1305564994_00513f9a5b.jpg#3
1305564994_00513f9a5b.jpg#4

A man in street racer armor be examine the tire of another
Two racer drive a white bike down a road .
Two motorist be ride along on their vehicle that be oddly
Two person be in a small race car drive by a green hill .
Two person in race uniform in a street car .

```

The file ExpertAnnotations.txt indicates which of the descriptions for each image were
written by experts which were written by crowdsource workers asked to describe the image.
Finally, the file CrowdFlowerAnnotations.txt provides the frequency of crowd workers that
indicate whether captions suit each image. These frequencies can be interpreted probabilistically.
The authors of the paper describe the annotations as follows:
... annotators were asked to write sentences that describe the depicted scenes,
situations, events and entities (people, animals, other objects). We collected multiple
captions for each image because there is a considerable degree of variance in the way
many images can be described.
— Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics, 2013.
There are also lists of the photo identifiers to use in a train/test split so that you can compare
results reported in the paper. The first step is to decide which captions to use. The simplest
approach is to use the first description for each photograph. First, we need a function to load
the entire annotations file (Flickr8k.token.txt) into memory. Below is a function to do this
called load doc() that, given a filename, will return the document as a string.

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

We can see from the sample of the file above that we need only split each line by white space
and take the first element as the image identifier and the rest as the image description. For
example:

```
# split line by white space
tokens = line.split()
# take the first token as the image id, the rest as the description
image_id, image_desc = tokens[0], tokens[1:]

```

We can then clean up the image identifier by removing the filename extension and the
description number.

```
# remove filename from image id
image_id = image_id.split('.')[0]

```

We can also put the description tokens back together into a string for later processing.

```
# convert description tokens back to string
image_desc = ' '.join(image_desc)

```

We can put all of this together into a function. Below defines the load descriptions()
function that will take the loaded file, process it line-by-line, and return a dictionary of image
identifiers to their first description.

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
# extract descriptions for images
def load_descriptions(doc):
mapping = dict()
# process lines
for line in doc.split('\n'):
# split line by white space
tokens = line.split()
if len(line) < 2:
continue
# take the first token as the image id, the rest as the description
image_id, image_desc = tokens[0], tokens[1:]
# remove filename from image id
image_id = image_id.split('.')[0]
# convert description tokens back to string
image_desc = ' '.join(image_desc)
# store the first description for each image
if image_id not in mapping:
mapping[image_id] = image_desc
return mapping
filename = 'Flickr8k_text/Flickr8k.token.txt'
doc = load_doc(filename)
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))

```

Running the example prints the number of loaded image descriptions.

```
Loaded: 8092

```

There are other ways to load descriptions that may turn out to be more accurate for the
data. Use the above example as a starting point and let me know what you come up with.

25.6

Prepare Description Text

The descriptions are tokenized; this means that each token is comprised of words separated by
white space. It also means that punctuation are separated as their own tokens, such as periods
('.') and apostrophes for word plurals ('s). It is a good idea to clean up the description text
before using it in a model. Some ideas of data cleaning we can form include:
- Normalizing the case of all tokens to lowercase.
- Remove all punctuation from tokens.
- Removing all tokens that contain one or fewer characters (after punctuation is removed),
e.g. 'a' and hanging 's' characters.

We can implement these simple cleaning operations in a function that cleans each description
in the loaded dictionary from the previous section. Below defines the clean descriptions()
function that will clean each loaded description.

```
# clean description text
def clean_descriptions(descriptions):
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
for key, desc in descriptions.items():
# tokenize
desc = desc.split()
# convert to lower case
desc = [word.lower() for word in desc]
# remove punctuation from each word
desc = [re_punc.sub('', w) for w in desc]
# remove hanging 's' and 'a'
desc = [word for word in desc if len(word)>1]
# store as string
descriptions[key] = ' '.join(desc)

```

We can then save the clean text to file for later use by our model. Each line will contain the
image identifier followed by the clean description. Below defines the save doc() function for
saving the cleaned descriptions to file.

```
# save descriptions to file, one per line
def save_doc(descriptions, filename):
lines = list()
for key, desc in mapping.items():
lines.append(key + ' ' + desc)
data = '\n'.join(lines)
file = open(filename, 'w')
file.write(data)
file.close()

```

Putting this all together with the loading of descriptions from the previous section, the
complete example is listed below.

```
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
# extract descriptions for images
def load_descriptions(doc):
mapping = dict()
# process lines
for line in doc.split('\n'):
# split line by white space
tokens = line.split()
if len(line) < 2:
continue
# take the first token as the image id, the rest as the description
image_id, image_desc = tokens[0], tokens[1:]
# remove filename from image id
image_id = image_id.split('.')[0]
# convert description tokens back to string
image_desc = ' '.join(image_desc)
# store the first description for each image
if image_id not in mapping:
mapping[image_id] = image_desc
return mapping
# clean description text
def clean_descriptions(descriptions):
# prepare regex for char filtering
re_punc = re.compile('[%s]' % re.escape(string.punctuation))
for key, desc in descriptions.items():
# tokenize
desc = desc.split()
# convert to lower case
desc = [word.lower() for word in desc]
# remove punctuation from each word
desc = [re_punc.sub('', w) for w in desc]
# remove hanging 's' and 'a'
desc = [word for word in desc if len(word)>1]
# store as string
descriptions[key] = ' '.join(desc)
# save descriptions to file, one per line
def save_doc(descriptions, filename):
lines = list()
for key, desc in descriptions.items():
lines.append(key + ' ' + desc)
data = '\n'.join(lines)
file = open(filename, 'w')
file.write(data)
file.close()
filename = 'Flickr8k_text/Flickr8k.token.txt'
# load descriptions
doc = load_doc(filename)
# parse descriptions
descriptions = load_descriptions(doc)
print('Loaded: %d ' % len(descriptions))
# clean descriptions
clean_descriptions(descriptions)
# summarize vocabulary
all_tokens = ' '.join(descriptions.values()).split()
vocabulary = set(all_tokens)
print('Vocabulary Size: %d' % len(vocabulary))
# save descriptions
save_doc(descriptions, 'descriptions.txt')

```

Running the example first loads 8,092 descriptions, cleans them, summarizes the vocabulary
of 4,484 unique words, then saves them to a new file called descriptions.txt.

```
Loaded: 8092
Vocabulary Size: 4484

```

Open the new file descriptions.txt in a text editor and review the contents. You should
see somewhat readable descriptions of photos ready for modeling.

```
...
3139118874_599b30b116 two girls pose for picture at christmastime
2065875490_a46b58c12b person is walking on sidewalk and skeleton is on the left inside of
fence
2682382530_f9f8fd1e89 man in black shorts is stretching out his leg
3484019369_354e0b88c0 hockey team in red and white on the side of the ice rink
505955292_026f1489f2 boy rides horse

```

The vocabulary is still relatively large. To make modeling easier, especially the first time
around, I would recommend further reducing the vocabulary by removing words that only
appear once or twice across all descriptions.

Whole Description Sequence Model

There are many ways to model the caption generation problem. One naive way is to create a
model that outputs the entire textual description in a one-shot manner. This is a naive model
because it puts a heavy burden on the model to both interpret the meaning of the photograph
and generate words, then arrange those words into the correct order.
This is not unlike the language translation problem used in an Encoder-Decoder recurrent
neural network where the entire translated sentence is output one word at a time given an
encoding of the input sequence. Here we would use an encoding of the image to generate the
output sentence instead. The image may be encoded using a pre-trained model used for image
classification, such as the VGG trained on the ImageNet model mentioned above.
The output of the model would be a probability distribution over each word in the vocabulary.
The sequence would be as long as the longest photo description. The descriptions would, therefore,
need to be first integer encoded where each word in the vocabulary is assigned a unique integer
and sequences of words would be replaced with sequences of integers. The integer sequences
would then need to be one hot encoded to represent the idealized probability distribution
over the vocabulary for each word in the sequence. We can use tools in Keras to prepare the
descriptions for this type of model. The first step is to load the mapping of image identifiers to
clean descriptions stored in descriptions.txt.

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
# load clean descriptions into memory
def load_clean_descriptions(filename):
doc = load_doc(filename)
descriptions = dict()
for line in doc.split('\n'):
# split line by white space
tokens = line.split()

# split id from description
image_id, image_desc = tokens[0], tokens[1:]
# store
mapping[image_id] = ' '.join(image_desc)
return descriptions
descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))

```

Running this piece loads the 8,092 photo descriptions into a dictionary keyed on image
identifiers. These identifiers can then be used to load each photo file for the corresponding
inputs to the model.

```
Loaded 8092

```

Next, we need to extract all of the description text so we can encode it.

```
# extract all text
desc_text = list(descriptions.values())

```

We can use the Keras Tokenizer class to consistently map each word in the vocabulary to
an integer. First, the object is created, then is fit on the description text. The fit tokenizer can
later be saved to file for consistent decoding of the predictions back to vocabulary words.

```
from keras.preprocessing.text import Tokenizer
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)

```

Next, we can use the fit tokenizer to encode the photo descriptions into sequences of integers.

```
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)

```

The model will require all output sequences to have the same length for training. We can
achieve this by padding all encoded sequences to have the same length as the longest encoded
sequence. We can pad the sequences with 0 values after the list of words. Keras provides the
pad sequences() function to pad the sequences.

```
from keras.preprocessing.sequence import pad_sequences
# pad all sequences to a fixed length
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)
padded = pad_sequences(sequences, maxlen=max_length, padding='post')

```

Finally, we can one hot encode the padded sequences to have one sparse vector for each word
in the sequence. Keras provides the to categorical() function to perform this operation.

```
from keras.utils import to_categorical
# one hot encode
y = to_categorical(padded, num_classes=vocab_size)

```

Once encoded, we can ensure that the sequence output data has the right shape for the
model.

```
y = y.reshape((len(descriptions), max_length, vocab_size))
print(y.shape)

```

Putting all of this together, the complete example is listed below.

```
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# load clean descriptions into memory
def load_clean_descriptions(filename):
doc = load_doc(filename)
descriptions = dict()
for line in doc.split('\n'):
# split line by white space
tokens = line.split()
# split id from description
image_id, image_desc = tokens[0], tokens[1:]
# store
descriptions[image_id] = ' '.join(image_desc)
return descriptions
descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
# extract all text
desc_text = list(descriptions.values())
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)
# pad all sequences to a fixed length

max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)
padded = pad_sequences(sequences, maxlen=max_length, padding='post')
# one hot encode
y = to_categorical(padded, num_classes=vocab_size)
y = y.reshape((len(descriptions), max_length, vocab_size))
print(y.shape)

```

Running the example first prints the number of loaded image descriptions (8,092 photos),
the dataset vocabulary size (4,485 words), the length of the longest description (28 words), then
finally the shape of the data for fitting a prediction model in the form [samples, sequence length,
features].

```
Loaded 8092
Vocabulary Size: 4485
Description Length: 28
(8092, 28, 4485)

```

As mentioned, outputting the entire sequence may be challenging for the model. We will
look at a simpler model in the next section.

Word-By-Word Model

A simpler model for generating a caption for photographs is to generate one word given both
the image as input and the last word generated. This model would then have to be called
recursively to generate each word in the description with previous predictions as input. Using
the word as input, give the model a forced context for predicting the next word in the sequence.
This is the model used in prior research, such as: Show and Tell: A Neural Image Caption
Generator, 2015. A word embedding layer can be used to represent the input words. Like the
feature extraction model for the photos, this too can be pre-trained either on a large corpus or
on the dataset of all descriptions.
The model would take a full sequence of words as input; the length of the sequence would be
the maximum length of descriptions in the dataset. The model must be started with something.
One approach is to surround each photo description with special tags to signal the start and
end of the description, such as STARTDESC and ENDDESC. For example, the description:
boy rides horse

```

Would become:
STARTDESC boy rides horse ENDDESC

```

And would be fed to the model with the same image input to result in the following
input-output word sequence pairs:

```
Input (X), Output (y)
STARTDESC, boy
STARTDESC, boy, rides
STARTDESC, boy, rides, horse
STARTDESC, boy, rides, horse ENDDESC

```

The data preparation would begin much the same as was described in the previous section.
Each description must be integer encoded. After encoding, the sequences are split into multiple
input and output pairs and only the output word (y) is one hot encoded. This is because the
model is only required to predict the probability distribution of one word at a time. The code is
the same up to the point where we calculate the maximum length of sequences.

```
...
descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
# extract all text
desc_text = list(descriptions.values())
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)
# determine the maximum sequence length
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)

```

Next, we split the each integer encoded sequence into input and output pairs. Let's step
through a single sequence called seq at the i'th word in the sequence, where i more than or
equal to 1. First, we take the first i-1 words as the input sequence and the i'th word as the
output word.

```
# split into input and output pair
in_seq, out_seq = seq[:i], seq[i]

```

Next, the input sequence is padded to the maximum length of the input sequences. Prepadding is used (the default) so that new words appear at the end of the sequence, instead of
the input beginning.

```
# pad input sequence
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]

```

The output word is one hot encoded, much like in the previous section.

```
# encode output sequence
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

```

We can put all of this together into a complete example to prepare description data for the
word-by-word model.

```
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# load clean descriptions into memory
def load_clean_descriptions(filename):
doc = load_doc(filename)
descriptions = dict()
for line in doc.split('\n'):
# split line by white space
tokens = line.split()
# split id from description
image_id, image_desc = tokens[0], tokens[1:]
# store
descriptions[image_id] = ' '.join(image_desc)
return descriptions
descriptions = load_clean_descriptions('descriptions.txt')
print('Loaded %d' % (len(descriptions)))
# extract all text
desc_text = list(descriptions.values())
# prepare tokenizer
tokenizer = Tokenizer()
tokenizer.fit_on_texts(desc_text)
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# integer encode descriptions
sequences = tokenizer.texts_to_sequences(desc_text)
# determine the maximum sequence length
max_length = max(len(s) for s in sequences)
print('Description Length: %d' % max_length)
X, y = list(), list()
for img_no, seq in enumerate(sequences):
# split one sequence into multiple X,y pairs
for i in range(1, len(seq)):
# split into input and output pair
in_seq, out_seq = seq[:i], seq[i]
# pad input sequence
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
# encode output sequence
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
# store
X.append(in_seq)
y.append(out_seq)
# convert to numpy arrays
X, y = array(X), array(y)
print(X.shape)
print(y.shape)

```

Running the example prints the same statistics, but prints the size of the resulting encoded
input and output sequences. Note that the input of images must follow the exact same ordering
where the same photo is shown for each example drawn from a single description. One way
to do this would be to load the photo and store it for each example prepared from a single
description.

```
Loaded 8092
Vocabulary Size: 4485
Description Length: 28
(66456, 28)
(66456, 4485)

```


Progressive Loading

The Flicr8K dataset of photos and descriptions can fit into RAM, if you have a lot of RAM
(e.g. 8 Gigabytes or more), and most modern systems do. This is fine if you want to fit a deep
learning model using the CPU. Alternately, if you want to fit a model using a GPU, then you
will not be able to fit the data into memory of an average GPU video card. One solution is to
progressively load the photos and descriptions as-needed by the model.
Keras supports progressively loaded datasets by using the fit generator() function on the
model. A generator is the term used to describe a function used to return batches of samples
for the model to train on. This can be as simple as a standalone function, the name of which is
passed to the fit generator() function when fitting the model. As a reminder, a model is fit
for multiple epochs, where one epoch is one pass through the entire training dataset, such as all
photos. One epoch is comprised of multiple batches of examples where the model weights are
updated at the end of each batch.
A generator must create and yield one batch of examples. For example, the average sentence
length in the dataset is 11 words; that means that each photo will result in 11 examples for
fitting the model and two photos will result in about 22 examples on average. A good default
batch size for modern hardware may be 32 examples, so that is about 2-3 photos worth of
examples.
We can write a custom generator to load a few photos and return the samples as a single
batch. Let's assume we are working with a word-by-word model described in the previous
section that expects a sequence of words and a prepared image as input and predicts a single
word. Let's design a data generator that given a loaded dictionary of image identifiers to clean
descriptions, a trained tokenizer, and a maximum sequence length will load one-image worth of
examples for each batch.

A generator must loop forever and yield each batch of samples. We can loop forever with
a while loop and within this, loop over each image in the image directory. For each image
filename, we can load the image and create all of the input-output sequence pairs from the
image's description. Below is the data generator function.

```
def data_generator(mapping, tokenizer, max_length):
# loop for ever over images
directory = 'Flicker8k_Dataset'
while 1:
for name in listdir(directory):
# load an image from file
filename = directory + '/' + name
image, image_id = load_image(filename)
# create word sequences
desc = mapping[image_id]
in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, image)
yield [[in_img, in_seq], out_word]

```

You could extend it to take the name of the dataset directory as a parameter. The generator
returns an array containing the inputs (X) and output (y) for the model. The input is comprised
of an array with two items for the input images and encoded word sequences. The outputs
are one hot encoded words. You can see that it calls a function called load photo() to load a
single photo and return the pixels and image identifier. This is a simplified version of the photo
loading function developed at the beginning of this tutorial.

```
# load a single photo intended as input for the VGG feature extractor model
def load_photo(filename):
image = load_img(filename, target_size=(224, 224))
# convert the image pixels to a NumPy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)[0]
# get image id
image_id = filename.split('/')[-1].split('.')[0]
return image, image_id

```

Another function named create sequences() is called to create sequences of images, input
sequences of words, and output words that we then yield to the caller. This is a function that
includes everything discussed in the previous section, and also creates copies of the image pixels,
one for each input-output pair created from the photo's description.

```
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, images):
Ximages, XSeq, y = list(), list(),list()
vocab_size = len(tokenizer.word_index) + 1
for j in range(len(descriptions)):
seq = descriptions[j]
image = images[j]
# integer encode
seq = tokenizer.texts_to_sequences([seq])[0]

# split one sequence into multiple X,y pairs
for i in range(1, len(seq)):
# select
in_seq, out_seq = seq[:i], seq[i]
# pad input sequence
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
# encode output sequence
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
# store
Ximages.append(image)
XSeq.append(in_seq)
y.append(out_seq)
Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
return Ximages, XSeq, y

```

Prior to preparing the model that uses the data generator, we must load the clean descriptions,
prepare the tokenizer, and calculate the maximum sequence length. All 3 of must be passed to
the data generator() as parameters. We use the same load clean descriptions() function
developed previously and a new create tokenizer() function that simplifies the creation of
the tokenizer. Tying all of this together, the complete data generator is listed below, ready for
use to train a model.

```
from os import listdir
from os import path
from numpy import array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

# load doc into memory
def load_doc(filename):
# open the file as read only
file = open(filename, 'r')
# read all text
text = file.read()
# close the file
file.close()
return text
# load clean descriptions into memory
def load_clean_descriptions(filename):
doc = load_doc(filename)
descriptions = dict()
for line in doc.split('\n'):
# split line by white space
tokens = line.split()
# split id from description
image_id, image_desc = tokens[0], tokens[1:]
# store
descriptions[image_id] = ' '.join(image_desc)
return descriptions
# fit a tokenizer given caption descriptions
def create_tokenizer(descriptions):
lines = list(descriptions.values())
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lines)
return tokenizer
# load a single photo intended as input for the VGG feature extractor model
def load_photo(filename):
image = load_img(filename, target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)[0]
# get image id
image_id = path.basename(filename).split('.')[0]
return image, image_id
# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, desc, image):
Ximages, XSeq, y = list(), list(),list()
vocab_size = len(tokenizer.word_index) + 1
# integer encode the description
seq = tokenizer.texts_to_sequences([desc])[0]
# split one sequence into multiple X,y pairs
for i in range(1, len(seq)):
# select
in_seq, out_seq = seq[:i], seq[i]
# pad input sequence
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
# encode output sequence
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
# store
Ximages.append(image)
XSeq.append(in_seq)
y.append(out_seq)
Ximages, XSeq, y = array(Ximages), array(XSeq), array(y)
return [Ximages, XSeq, y]
# data generator, intended to be used in a call to model.fit_generator()
def data_generator(descriptions, tokenizer, max_length):
# loop for ever over images
directory = 'Flicker8k_Dataset'
while 1:
for name in listdir(directory):
# load an image from file
filename = path.join(directory, name)
image, image_id = load_photo(filename)
# create word sequences
desc = descriptions[image_id]
in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc, image)
yield [[in_img, in_seq], out_word]
# load mapping of ids to descriptions
descriptions = load_clean_descriptions('descriptions.txt')
# integer encode sequences of words
tokenizer = create_tokenizer(descriptions)
# pad to fixed length
max_length = max(len(s.split()) for s in list(descriptions.values()))
print('Description Length: %d' % max_length)
# test the data generator
generator = data_generator(descriptions, tokenizer, max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)

```

A data generator can be tested by calling the next() function. We can test the generator as
follows.

```
# test the data generator
generator = data_generator(descriptions, tokenizer, max_length)
inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)

```

Running the example prints the shape of the input and output example for a single batch
(e.g. 13 input-output pairs):

```
(13, 224, 224, 3)
(13, 28)
(13, 4485)

```

The generator can be used to fit a model by calling the fit generator() function on the
model (instead of fit()) and passing in the generator. We must also specify the number of
steps or batches per epoch. We could estimate this as (10 x training dataset size), perhaps
70,000 if 7,000 images are used for training.

```
# define model
# ...
# fit model
model.fit_generator(data_generator(descriptions, tokenizer, max_length),
steps_per_epoch=70000, ...)

```

##### Run Notebook
Click notebook `01_load_photos.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `02_pre_calculate_features.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `03_load_descriptions.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `04_clean_descriptions.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `05_whole_description_model.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `06_word_by_word.ipynb` in jupterLab UI and run jupyter notebook.

##### Run Notebook
Click notebook `07_progressive_loading.ipynb` in jupterLab UI and run jupyter notebook.

### Further Reading

This section provides more resources on the topic if you are looking go deeper.

Flickr8K Dataset

- Framing image description as a ranking task: data, models and evaluation metrics (Homepage).
http://nlp.cs.illinois.edu/HockenmaierGroup/Framing_Image_Description/KCCA.
html
- Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics,
013.
https://www.jair.org/media/3994/live-3994-7274-jair.pdf
- Dataset Request Form.
https://illinois.edu/fb/sec/1713398
- Old Flicrk8K Homepage.
http://nlp.cs.illinois.edu/HockenmaierGroup/8k-pictures.html

API

- Python Generators.
https://wiki.python.org/moin/Generators
- Keras Model API.
https://keras.io/models/model/
- Keras pad sequences() API.
https://keras.io/preprocessing/sequence/#pad_sequences
- Keras Tokenizer API.
https://keras.io/preprocessing/text/#tokenizer
- Keras VGG16 API.
https://keras.io/applications/#vgg16

##### Summary

In this tutorial, you discovered how to prepare photos and textual descriptions ready for
developing an automatic photo caption generation model. Specifically, you learned:
- About the Flickr8K dataset comprised of more than 8,000 photos and up to 5 captions for
each photo.
- How to generally load and prepare photo and text data for modeling with deep learning.
- How to specifically encode data for two different types of deep learning models in Keras.

