# Importing Modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the Dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting= 3)

import re 
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    #Cleaning the texts
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # Making all character small
    review = review.lower()
    # Splitting the string
    review = review.split()
    # Importing useless words list and cleaning the string
    # Stemming 
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()

#Dependent Variable 
y = dataset.iloc[:,1].values

#Splitting the dataset into Training and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

#Fitting Classifier to the Training set
#create our classifier here
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#Predicting Single Value/ new result with regression
y_pred = classifier.predict(X_test)

#Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)