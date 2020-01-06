# How to Load and Use a Pre-Trained Object Recognition Model
Convolutional neural networks are now capable of outperforming humans on some computer
vision tasks, such as classifying images. That is, given a photograph of an object, answer the
question as to which of 1,000 specific objects the photograph shows. A competition-winning
model for this task is the VGG model by researchers at Oxford. What is important about this
model, besides its capability of classifying objects in photographs, is that the model weights
are freely available and can be loaded and used in your own models and applications. In this
tutorial, you will discover the VGG convolutional neural network models for image classification.
After completing this tutorial, you will know:
- About the ImageNet dataset and competition and the VGG winning models.
- How to load the VGG model in Keras and summarize its structure.
- How to use the loaded VGG model to classifying objects in ad hoc photographs.

Let's get started.

23.1

Tutorial Overview

This tutorial is divided into the following parts:
1. ImageNet
2. The Oxford VGG Models
3. Load the VGG Model in Keras
4. Develop a Simple Photo Classifier
Note, Keras makes use of the Python Imaging Library or PIL library for manipulating
images. Installation on your system may vary.

ImageNet

ImageNet is a research project to develop a large database of images with annotations, e.g.
images and their descriptions. The images and their annotations have been the basis for an
image classification challenge called the ImageNet Large Scale Visual Recognition Challenge
or ILSVRC since 2010. The result is that research organizations battle it out on pre-defined
datasets to see who has the best model for classifying the objects in images.
The ImageNet Large Scale Visual Recognition Challenge is a benchmark in object
category classification and detection on hundreds of object categories and millions
of images. The challenge has been run annually from 2010 to present, attracting
participation from more than fifty institutions.
— ImageNet Large Scale Visual Recognition Challenge, 2015.
For the classification task, images must be classified into one of 1,000 different categories.
For the last few years very deep convolutional neural network models have been used to win
these challenges and results on the tasks have exceeded human performance.

![](./274-25.png)

Taken From ImageNet Large Scale Visual Recognition Challenge.

The Oxford VGG Models

Researchers from the Oxford Visual Geometry Group, or VGG for short, participate in the
ILSVRC challenge. In 2014, convolutional neural network models (CNN) developed by the
VGG won the image classification tasks.

![](./275-26.png)

After the competition, the participants wrote up their findings in the paper Very Deep
Convolutional Networks for Large-Scale Image Recognition, 2014. They also made their models
and learned weights available online. This allowed other researchers and developers to use a
state-of-the-art image classification model in their own work and programs. This helped to fuel
a rash of transfer learning work where pre-trained models are used with minor modification
on wholly new predictive modeling tasks, harnessing the state-of-the-art feature extraction
capabilities of proven models.
... we come up with significantly more accurate ConvNet architectures, which not
only achieve the state-of-the-art accuracy on ILSVRC classification and localisation
tasks, but are also applicable to other image recognition datasets, where they achieve
excellent performance even when used as a part of a relatively simple pipelines (e.g.
deep features classified by a linear SVM without fine-tuning). We have released our
two best-performing models to facilitate further research.
— Very Deep Convolutional Networks for Large-Scale Image Recognition, 2014.
VGG released two different CNN models, specifically a 16-layer model and a 19-layer model.
Refer to the paper for the full details of these models. The VGG models are not longer state-ofthe-art by only a few percentage points. Nevertheless, they are very powerful models and useful
both as image classifiers and as the basis for new models that use image inputs. In the next
section, we will see how we can use the VGG model directly in Keras.

23.4

Load the VGG Model in Keras

The VGG model can be loaded and used in the Keras deep learning library. Keras provides an
Applications interface for loading and using pre-trained models. Using this interface, you can
create a VGG model using the pre-trained weights provided by the Oxford group and use it as
a starting point in your own model, or use it as a model directly for classifying images. In this
tutorial, we will focus on the use case of classifying new images using the VGG model. Keras
provides both the 16-layer and 19-layer version via the VGG16 and VGG19 classes. Let's focus
on the VGG16 model. The model can be created as follows:

```
from keras.applications.vgg16 import VGG16
model = VGG16()
```

That's it. The first time you run this example, Keras will download the weight files from
the Internet and store them in the ∼/.keras/models directory. Note that the weights are
about 528 megabytes, so the download may take a few minutes depending on the speed of your
Internet connection.
The weights are only downloaded once. The next time you run the example, the weights are
loaded locally and the model should be ready to use in seconds. We can use the standard Keras
tools for inspecting the model structure. For example, you can print a summary of the network
layers as follows:

```
from keras.applications.vgg16 import VGG16
model = VGG16()
model.summary()
```

You can see that the model is huge. You can also see that, by default, the model expects
images as input with the size 224 x 224 pixels with 3 channels (e.g. color).

```
_________________________________________________________________
Layer (type)
Output Shape
Param #
=================================================================
input_1 (InputLayer)
(None, 224, 224, 3)
0
_________________________________________________________________
block1_conv1 (Conv2D)
(None, 224, 224, 64)
1792
_________________________________________________________________
block1_conv2 (Conv2D)
(None, 224, 224, 64)
36928
_________________________________________________________________
block1_pool (MaxPooling2D) (None, 112, 112, 64) 0
_________________________________________________________________
block2_conv1 (Conv2D)
(None, 112, 112, 128) 73856
_________________________________________________________________
block2_conv2 (Conv2D)
(None, 112, 112, 128) 147584
_________________________________________________________________
block2_pool (MaxPooling2D) (None, 56, 56, 128)
0
_________________________________________________________________
block3_conv1 (Conv2D)
(None, 56, 56, 256)
295168
_________________________________________________________________
block3_conv2 (Conv2D)
(None, 56, 56, 256)
590080
_________________________________________________________________
block3_conv3 (Conv2D)
(None, 56, 56, 256)
590080
_________________________________________________________________
block3_pool (MaxPooling2D) (None, 28, 28, 256)
0
_________________________________________________________________
block4_conv1 (Conv2D)
(None, 28, 28, 512)
1180160
_________________________________________________________________

block4_conv2 (Conv2D)
(None, 28, 28, 512)
2359808
_________________________________________________________________
block4_conv3 (Conv2D)
(None, 28, 28, 512)
2359808
_________________________________________________________________
block4_pool (MaxPooling2D) (None, 14, 14, 512)
0
_________________________________________________________________
block5_conv1 (Conv2D)
(None, 14, 14, 512)
2359808
_________________________________________________________________
block5_conv2 (Conv2D)
(None, 14, 14, 512)
2359808
_________________________________________________________________
block5_conv3 (Conv2D)
(None, 14, 14, 512)
2359808
_________________________________________________________________
block5_pool (MaxPooling2D) (None, 7, 7, 512)
0
_________________________________________________________________
flatten (Flatten)
(None, 25088)
0
_________________________________________________________________
fc1 (Dense)
(None, 4096)
102764544
_________________________________________________________________
fc2 (Dense)
(None, 4096)
16781312
_________________________________________________________________
predictions (Dense)
(None, 1000)
4097000
=================================================================
Total params: 138,357,544
Trainable params: 138,357,544
Non-trainable params: 0
_________________________________________________________________
```

We can also create a plot of the layers in the VGG model, as follows:

```
from keras.applications.vgg16 import VGG16
from keras.utils.vis_utils import plot_model
model = VGG16()
plot_model(model, to_file='vgg.png')
```

Again, because the model is large, the plot is a little too large and perhaps unreadable.
Nevertheless, it is provided below.

##### Load the VGG Model in Keras

![](./278-27.png)

The VGG() class takes a few arguments that may only interest you if you are looking to use
the model in your own project, e.g. for transfer learning. For example:
- include top (True): Whether or not to include the output layers for the model. You
don't need these if you are fitting the model on your own problem.
- weights ('imagenet'): What weights to load. You can specify None to not load pretrained weights if you are interested in training the model yourself from scratch.
- input tensor (None): A new input layer if you intend to fit the model on new data of a
different size.
- input shape (None): The size of images that the model is expected to take if you change
the input layer.
- pooling (None): The type of pooling to use when you are training a new set of output
layers.
- classes (1000): The number of classes (e.g. size of output vector) for the model.

Next, let's look at using the loaded VGG model to classify ad hoc photographs.

# Develop a Simple Photo Classifier

Let's develop a simple image classification script.

### Get a Sample Image

First, we need an image we can classify. You can download a random photograph of a coffee
mug from Flickr.

![](./279-28.png)

Download the image and save it to your current working directory with the filename mug.jpg.


### Load the VGG Model

Load the weights for the VGG-16 model, as we did in the previous section.

```
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()
```


### Load and Prepare Image

Next, we can load the image as pixel data and prepare it to be presented to the network. Keras
provides some tools to help with this step. First, we can use the load img() function to load
the image and resize it to the required size of 224 x 224 pixels.

```
from keras.preprocessing.image import load_img
# load an image from file
image = load_img('mug.jpg', target_size=(224, 224))
```

Next, we can convert the pixels to a NumPy array so that we can work with it in Keras. We
can use the img to array() function for this.

```
from keras.preprocessing.image import img_to_array
# convert the image pixels to a NumPy array
image = img_to_array(image)
```

The network expects one or more images as input; that means the input array will need to
be 4-dimensional: samples, rows, columns, and channels. We only have one sample (one image).
We can reshape the array by calling reshape() and adding the extra dimension.

```
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
```

Next, the image pixels need to be prepared in the same way as the ImageNet training data
was prepared. Specifically, from the paper:
The only preprocessing we do is subtracting the mean RGB value, computed on the
training set, from each pixel.
— Very Deep Convolutional Networks for Large-Scale Image Recognition, 2014.
Keras provides a function called preprocess input() to prepare new input for the network.

```
from keras.applications.vgg16 import preprocess_input
# prepare the image for the VGG model
image = preprocess_input(image)
```

We are now ready to make a prediction for our loaded and prepared image.

##### Make a Prediction

We can call the predict() function on the model in order to get a prediction of the probability
of the image belonging to each of the 1,000 known object types.

```
# predict the probability across all output classes
yhat = model.predict(image)
```

Nearly there, now we need to interpret the probabilities.

##### Interpret Prediction

Keras provides a function to interpret the probabilities called decode predictions(). It can
return a list of classes and their probabilities in case you would like to present the top 3 objects
that may be in the photo. We will just report the first most likely object.

```
from keras.applications.vgg16 import decode_predictions
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
```

And that's it.

##### Complete Example

Tying all of this together, the complete example is listed below:

```
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.applications.vgg16 import VGG16
# load the model
model = VGG16()
# load an image from file
image = load_img('mug.jpg', target_size=(224, 224))
# convert the image pixels to a numpy array
image = img_to_array(image)
# reshape data for the model
image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
# prepare the image for the VGG model
image = preprocess_input(image)
# predict the probability across all output classes
yhat = model.predict(image)
# convert the probabilities to class labels
label = decode_predictions(yhat)
# retrieve the most likely result, e.g. highest probability
label = label[0][0]
# print the classification
print('%s (%.2f%%)' % (label[1], label[2]*100))
```

Running the example, we can see that the image is correctly classified as a coffee mug with
a 75% likelihood.

##### Run Notebook
Click notebook `1_classify_image.ipynb` in jupterLab UI and run jupyter notebook.

**Note:**  Given the stochastic nature of neural networks, your specific results may vary. Consider
running the example a few times.

```
coffee_mug (75.27%)
```

# Further Reading

This section provides more resources on the topic if you are looking go deeper.
- ImageNet.
http://www.image-net.org/
- ImageNet on Wikipedia.
https://en.wikipedia.org/wiki/ImageNet
- Very Deep Convolutional Networks for Large-Scale Image Recognition, 2015.
https://arxiv.org/abs/1409.1556
- Very Deep Convolutional Networks for Large-Scale Visual Recognition, at Oxford.
http://www.robots.ox.ac.uk/~vgg/research/very_deep/
- Building powerful image classification models using very little data, 2016.
https://blog.keras.io/building-powerful-image-classification-models-using-very-litt
html
- Keras Applications API.
https://keras.io/applications/
- Keras weight files files.
https://github.com/fchollet/deep-learning-models/releases/

# Summary

In this tutorial, you discovered the VGG convolutional neural network models for image
classification. Specifically, you learned:
- About the ImageNet dataset and competition and the VGG winning models.
- How to load the VGG model in Keras and summarize its structure.
- How to use the loaded VGG model to classifying objects in ad hoc photographs.
