# scrapping text from website to match to the queary from user to that the chatbot found from the makers website.

# import libraries
from newspaper import Article
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
import warnings

#ignore any warnings
warnings.filterwarnings('ignore')

#download packages from NLTK
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

article = Article('https://blog.makersacademy.com/tips-to-max-your-learning-5e363517acfb', 'https://blog.makersacademy.com/the-makers-academy-apprenticeship-application-process-d0834b614ed0')
article.download()
article.parse()
article.nlp()
corpus = article.text

# ££££££print(corpus)

#Tokenization

text = corpus
sent_tokens = nltk.sent_tokenize(text) #convert the text into a list of sentences
#print the list of sentences
# ££££££print(sent_tokens)

#creating a dictionary (key:value) pair to remove punctuations
#removing the punctuations
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)
# £££££££print(string.punctuation)

#print the dictionary
# ££££print(remove_punct_dict)

#create a list of lemmatised lowercase wordsafter removing punctuations
def LemNormalize(text):
    return nltk.word_tokenize(text.lower().translate(remove_punct_dict))


# £££££print(LemNormalize(text))

#Keyword Matching

# #Greating Inputs
# GREETING_INPUTS = ["hello", "hi", "greetings", "wassup", "hey"]

# #Greeting repsonse back to the user
# GREETING_RESPONSE=["hello", "hi", "greetings my firend", "wassup friend", "howdy", "hey there", "Hey, hope your well today"]

# def greeting(sentence):
#     #if user's input is a greting, then a randomly chosen greeting reponse will be give
#     for word in sentence.split():
#         if word.lower() in GREETING_INPUTS:
#             return random.choice(GREETING_RESPONSE)

#     # Greating Inputs

def response(user_response):

    user_response = user_response.lower()

    robo_response = ''

    #apending the user repsosne to the end of the blog's sentence list
    sent_tokens.append(user_response)
    # ££££print(sent_tokens)

    #create a Tfidf VectoriseObject
    TfidfVec = TfidfVectorizer(tokenizer=LemNormalize, stop_words='english')
    # TF -  idf -  have two different functions the two values are bing multipled together.

    #converts the text to a matrix of TF IDF features
    tfidf = TfidfVec.fit_transform(sent_tokens)
    # ££££print(tfidf)

    #get the similarity scorce = the website compare to the repsonse of the user
    vals = cosine_similarity(tfidf[-1], tfidf) #comparing the appended text[-1] to all of the text that was converted into a matrix
    #vals is a list of lists

    # print(vals) #this is prints the scores

    #get the index of the most similar text in the blog/sentence list to the user response in the question asked to the chatbot
    idx = vals.argsort()[0][-2] #vals is a list of lists, hence you need to be able to go into the first list tor then
    #[-1 is the one at the end of the list] [-2] the next best sentence is here, which is the sentence list it 'self

    #reduce the dimensonality of vals
    flat = vals.flatten()
    # print(flat)

    #sort the list in accending order
    flat = np.sort(flat)
    # print(flat)

    #get the similar scorce to the users response
    score = flat[-2]
    # print(score)

    #if the variable score is 0 then there is not text similar to the users response
    if(score == 0):
        #
        robo_response = robo_response + " SEARCH ERROR: I apologise, I didn't find what you were looking for."
    else:
        robo_response = robo_response+sent_tokens[idx]
        sent_tokens.remove(user_response)
        # print(robo_response)
    return robo_response


# flag = True
# print("MakersBot: I am Makers Bot or Tipy for short. I will answer your queries about Makers Academy on Optimising your learning. If you want to exit, type Bye!")
# while(flag == True):
#   user_response = input()
#   user_response = user_response.lower()
#   if(user_response != 'bye'):
#     if(user_response == 'thanks' or user_response =='thank you'):
#       flag=False
#       print("MakersBot: You are welcome, we wish you the best at Makers !")
#     else:
#       if(greeting(user_response) != None):
#         print("MakersBot: "+greeting(user_response))
#       else:
#         print("MakersBot: "+response(user_response))
#   else:
#     flag = False
#     print("MakersBot: Chat with you later, see you soon !")
