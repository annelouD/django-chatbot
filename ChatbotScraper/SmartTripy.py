from . import Scraper
import nltk
from datetime import datetime
import numpy
import tflearn
import tensorflow
import random
import json
import pickle
import warnings
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

#ignore any warnings
warnings.filterwarnings('ignore')

# with open("./intents.json") as file:
global data
data = json.load(open("/Users/ands/Desktop/MAKERS/final_proj/TriPy/TriPyChatbot/TriPyChatbot/ChatbotScraper/intents.json"))
# print(data)
# print(data['intents'][3]['patterns'])

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
    # rb is read bytes as this is what he file is saved as. Then it will load in the files. f is a file that will be created.
    # we are opening the save data
except:
    words = []
    labels = []
    doc_x = []
    doc_y = []

    # you can print the data to see the content by the following - to allow you to call the content of the array
    print(data["intents"])

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            # here we will be stemming to eb able to gain basic functionality of the machine learning for AI.
            # stemming takes each word and gives you the root! so rather than whats --> what or there --> there. as you don't care about the other things attached to the word. to not letter the model stray away from what is intended.
            # Tokenise gets all the words in the pattern, it splits by a space to get all the different words. it creates a list of words.
            words.extend(wrds)
            # because nltk.word is already a list you can just extend the array to add rather than append each individual word.
            doc_x.append(wrds)
            doc_y.append(intent["tag"])
            # for each pattern to correspond with its tag in the json file is necessary for the classify each pattern. This is important for training the model.

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    # PART 2
    # this is the pre-processing of the data
    #  we are going to stem all of the words in the list that has been created. As well as remove any duplicate elements
    # we want to know how many words that the chat bot has seen already.
    words = [stemmer.stem(w.lower()) for w in words if w != "?"]
    # N.B. if w not in "?" is the same things as if w != "?"
    # the if statment is to remove any ? that the user may type,as this is quite common.
    #This will remove all the duplicates in the words list, it then forces them into lowercase for the words in the words list
    words = sorted(list(set(words)))
    #we sort the list, set removes all the duplicates set is it's own data type, so list converts it back into ta list'


    labels = sorted(labels)


    # Training and testing output!
    # A neural network only understands numbers and not strings, hence the words need to be converted in to an array of numbers.
    # This is point we need to create a bag(array) of words that can be used to train the model.
    # a bag of words is one hot encoded.
    training = []
    # This is for th bag of words the chatbot to use with the sorted data, that is the conversion of ones and zeros.
    output = []
    # This is to understand the inversion of the numbers in words

    out_empty = [0 for _ in range (len(labels))]
    # The words inputed and worked out in ones & zeros will be whether they appear in the tags - this is what is represented by the labels variable

    for x, doc in enumerate(doc_x):
        bag = []

        wrdsPatt = [stemmer.stem(w) for w in doc]
    # We are stemming the words in doc x, and that the words exists in the current pattern that we are looping through.

        for w in words:
            #here we are looping through this ...words = sorted(list(set(words)))
            if w in wrdsPatt:
                bag.append(1)
                #When this word exists it is appended to the array, it only appears as a 1 if the words exists
            else:
                bag.append(0)
    # in order to push something through training array we need to do the following
            output_row = out_empty[:]
            output_row[labels.index(doc_y[x])] = 1
    # out_empty[:]  makes a copy of this code that was displayed before
    # we are looking through the labels(array) list and where the tag is and setting the value to 1 in the output rom

            training.append(bag)
            output.append(output_row)
    # these are both one hot encoded.

    # until this point we are getting our data ready to feed into our model.

#PART 3
# Making predictions based on a string of text
# We are going to change our training array into numpy arrays and forms.
    training = numpy.array(training)
    output = numpy.array(output)
    # we need to change output and our training & output into numpy arrays so that they can be feed to the model
    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

        #we need to save our try method that loads the arrays of the lists created into a file.

# From here the below code is where our AI aspect is created and drawn up and we are developing the model.
# This is where we are able to defined the architecture of our model.
# The numbers in the fully connected lines of code can be changed (trial and error) in creating a model powerful model
tensorflow.reset_default_graph()
# This will enable us to get rid of pre set, we get to reset underline graph data

net = tflearn.input_data(shape=[None, len(training[0])])
# this len(training) is the length of our training model
# this will help us to define the model. Tt will expect us to define the array.
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
# these are neurons tht are connected to the output of the labels
net = tflearn.regression(net)
# the fully connected layer is added to our neural network which is spaced above, but the input of data.
# there are 8 neurons that are created for the hidden layer, it is copied again
# this will allow us to get probability of out data for the layer that we have created.
model = tflearn.DNN(net)
# this model allows us to train it
# the AI elements and connects are finished here.
# The training starts here! ans we are passing it all the training data
#££££££££££££
try:
    model.load("model.tflearn")
    # we are opening the save data
except:
    model.fit(training, output, n_epoch=1000, batch_size=10, show_metric=True)
    model.save("model.tflearn")
#(*4mins)
# The epoch number the number of times that it will seee the same data can be changed to increase the number of replications, it is all trial and error.
# The put in the model 1000 replications can the more it see & understands the words inputted in the better it is at giving the desired response.
# The show metric will generate an nice output in terminal when we are generating the model.
# There are a few things hat we need to do to get the model running.
# We need to make our process more efficient and don't want to have to run the code above every single time we wish to run and training the model from starch, as every time we wish to make one predication in our current state.
# We need to save the model in a different and better way. we don't want to have to do the preprocessing of this data multiple time.
# Hence we added a try/accept clause into out code to be able to do this!
# note now that the data is saved to the folder, if there are any changes, please note that the files need to be deleted or renamed.

#PART 4
def bag_of_words(s, woords):
    bag_w = [0 for _ in range(len(woords))]

    se_words = nltk.word_tokenize(s)
    se_words = [stemmer.stem(woords.lower()) for woords in se_words]

    for se in se_words:
        for i, w in enumerate(woords):
            if w == se:
                bag_w[i] = 1

    return numpy.array(bag_w)

def tripy_timestamp():
        time_now = datetime.now()
        return time_now

def chat():
    # global data
    print("Hello there, Start talking with me, Tripy, the ChatBot!")
    name = input("Please give me your name \n")
    print(random.choice(data["intents"][0]["responses"]) + name + "\n When you are finished with me type: Bye, Goodbye, Cya ! \n Ask me a question about myself or Makers?")
    global flag
    flag = True
    while(flag == True):
        inp = input(f"{name}: ")
        if inp.lower() == "bye" or inp.lower() == "goodbye" or inp.lower() == "cya":
            flag=False
            print("ChatBot - Tripy: " + random.choice(data["intents"][3]["patterns"]) + "\n You are welcome, I wish you the best of luck in the future! " + "\n ~ END OF COMMUNITCATION ~")
            break

        if inp.lower() == "tripy tell me the time" or inp.lower() == "what is the time?" or inp.lower() =="what is the time tripy" or inp.lower() == "what is the time" or inp.lower() == "Tell me the time please?":
            print(tripy_timestamp())
            continue
        
        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        # print(results_index)
        # print(results)
        # print(results[0][results_index])
        # print(tag)

# To include the scraper need to be based on probability.
        if results[0][results_index] > 0.999:
    #tech/w tim code if results[results_index] > 0.7:  this does not work for me  
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses1 = tg['responses']

            print("Tripy:" + random.choice(responses1))
        else:
            print("Tripy: " + response(inp))


# the scraper will need to be added to the else, based on an imported file's method, see below.
# print("ChatBot - Tripy: " + scraper_response(inp))


# The chat function handles getting a prediction from the model (tensorflow) and grabbing an appropriate response from our JSON file of responses.
# chat()
