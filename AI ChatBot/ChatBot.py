import nltk
from nltk.stem.lancaster import LancasterStemmer
nltk.download('punkt')

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle


# Loading in our json file as "data"; pretty much a python dictionary
with open("intents.json") as file:
    data = json.load(file)

# Try to open up some saved data, and if we can't then go through saving it
try:
    with open("data.pickle", "rb") as f:
        # Save our four data values into the pickle file and load it
        words, labels, training, output = pickle.load(f)
except:
    # Just uses the main tag intents to call the rest of the tags in the json file
    # print(data["intents"])
    # Looping through the json data and inserting the respective groups
    words = []
    labels = []
    # Have these two so docs_x takes all of the docs but docs_y can take all of the tags that those docs are part of
    # respectively
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        # Known as stemming: Basically, we take each word in the pattern and bring it down to the root word (no
        # punctuations or the s at the end and etc. as we don't care about these extra patterns). To stem the words we
        # need to tokenize them or get each individual word (like splitting by spaces).
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            # Since, nltk.word_tokenize(pattern) is a list we can just extend the words list to include wrds
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

            if intent["tag"] not in labels:
                # Adds all of the tags into labels
                labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    # Set removes all of the duplicate words and list converts it back to a list cause set is a data type and then we
    # just sort it
    words = sorted(list(set(words)))
    # Sorting labels
    labels = sorted(labels)
    # We are going to create a bag of words because what we've just created won't work for the neural network as that
    # requires integers as we've just created a string of words. Essentially, we are going to put the whether or not the
    # word exists in our vocabulary of the words in a list (this is called one-hot encoded) (TU: 4:54:43) TU: 4:55:30
    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]
    # So, for our output, if a tag exists then we will make it a 1 rather than it be a 0
    for x, doc in enumerate(docs_x):
        bag = []
        wrds = [stemmer.stem(w) for  w in doc]
        # TU: 4:58:01
        # So, we are going to put into bag a 1 or 0 if the word is in the main words list
        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
        # TU: 5:00:32
        output_row = out_empty[:]
        # Look through the labels and see where that tag is and set that position to 1
        output_row[labels.index(docs_y[x])] = 1
        # We have created two new lists; training has a bag of words and output has 0s and 1s
        training.append(bag)
        output.append(output_row)

    # Making them numpy arrays cause it is easier to do so
    training = numpy.array(training)
    output = numpy.array(output)
    with open("data.pickle", "wb") as f:
        # Write all of these pickle files and then save it into that pickle file so we can finally have some saved
        # values
        pickle.dump((words, labels, training, output), f)
# To make sure to get rid of all of the previous data graphs
tensorflow.compat.v1.reset_default_graph()
# TU: 5:05:35
# Making the neuron layers
# We are setting the expected shape of the input data by using len of the first training data (this is the first layer)
net = tflearn.input_data(shape=[None, len(training[0])])
# Add 8 neurons to the first hidden layer
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
# Softmax gives us a probability for each layers
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
# Training the dataset (DNN is a type of neural network)
model = tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    # TU how training and the neurons work, start at 5:08:29. Essentially, our input is taken through the neuron layers
    # and then the computer decides whichever tag has the most probability of being asked (if need to respond with a
    # hello tag and so forth). Fitting the model; essentially, passing all of our data to the model
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")

# Time to start making predictions (TU: 5:19:29)
# First we are going to turn the user's input into a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        # TU: 5:21:50
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

# We are going to create a function that takes in the user's input
def chat():
    print("Start talking with me! (Type quit to stop)")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break
        # TU: 5:24:19
        # Essentially, we add [0] to the below as it picks from a list of list the first list which is the results we need
        results = model.predict([bag_of_words(inp, words)])[0]
        # Gives us the index of the greatest probability
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        # Add the threshold for the results
        if results[results_index] > 0.7:
            # TU the question, why can't we go straight into the response rather than the tags: 5:28:00
            for tg in data["intents"]:
                if tg["tag"] == tag:
                    responses = tg['responses']
            print(random.choice(responses))
        else:
            print("I didn't get that, please type in another question!")
# Just outputs the probable of each neuron which isn't useful to us, so we need to up date it
        #print(results)

chat()

