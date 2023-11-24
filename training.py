import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Download necessary NLTK resources
nltk.download('wordnet')
nltk.download('punkt')

# Initialize WordNet lemmatizer
lemmatizer = WordNetLemmatizer()

# Load intents from JSON file
intents = json.loads(open('intents.json').read())

# Initialize lists for words, classes, and documents
words = []
classes = []
documents = []
ignore_letters = ['?', '!', ',', '.']

# Iterate through intents and patterns to preprocess data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize words in each pattern
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        # Create a tuple of tokenized words and the corresponding intent tag
        documents.append((word_list, intent['tag']))
        # Add the intent tag to the classes list if not already present
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lemmatize words and remove ignored letters
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]
# Remove duplicates and sort the words
words = sorted(set(words))
# Sort the classes
classes = sorted(set(classes))

# Save the words and classes to pickle files for later use
pickle.dump(words, open('words.pk1', 'wb'))
pickle.dump(classes, open('classes.pk1', 'wb'))

# Initialize an empty list for the output representation of intents
output_empty = [0] * len(classes)

# Initialize training data
training = []

# Iterate through documents to create a bag of words representation
for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    # Create a bag of words with 1s and 0s
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    # Create the output row with 1 at the index corresponding to the intent tag
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    # Append the bag of words and output row to the training data
    training.append([bag, output_row])

# Convert training data to NumPy arrays
train_x = np.array([item[0] for item in training])
train_y = np.array([item[1] for item in training])

# Build the neural network model using Keras
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Configure the Stochastic Gradient Descent optimizer
sgd = SGD(lr=0.01, momentum=0.9, nesterov=True)

# Compile the model with categorical crossentropy loss and SGD optimizer
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Train the model with the training data
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Save the trained model to a file
model.save('chatbot_model.h5', hist)

print('Done')
