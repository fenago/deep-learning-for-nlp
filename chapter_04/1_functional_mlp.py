# %%
'''
## Multilayer Perceptron
In this section, we define a Multilayer Perceptron model for binary classification. The model
has 10 inputs, 3 hidden layers with 10, 20, and 10 neurons, and an output layer with 1 output.
Rectified linear activation functions are used in each hidden layer and a sigmoid activation
function is used in the output layer, for binary classification.
'''

# %%
# example of a multilayer perceptron
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
visible = Input(shape=(10,))
hidden1 = Dense(10, activation='relu')(visible)
hidden2 = Dense(20, activation='relu')(hidden1)
hidden3 = Dense(10, activation='relu')(hidden2)
output = Dense(1, activation='sigmoid')(hidden3)
model = Model(inputs=visible, outputs=output)

# %%
''' 
Running the example prints the structure of the network.
'''

# %%
# summarize layers
model.summary()
# plot graph
plot_model(model, to_file='multilayer_perceptron_graph.png')

# %%
from PIL import Image
from IPython.display import display # to display images

image = Image.open('multilayer_perceptron_graph.png')
display(image)