# Ham-or-Spam
In this project I have used Naïve Bayes model which is a technique dependent on Bayes' Theorem with an assumption of freedom among predictors. In straightforward terms, a
Naive Bayes classifier expects that the occurrence of a specific component in a class is not related to the occurrence of some other element. This classifier uses Bayesian theorem.

How it works:
First we are importing stopwords from nltk library to remove stopwords from the dataset which makes the data more relevant for testing. Then we are importing CountVectorizer and 
using it to get the frequency of repeatedly occurring words, these words need to then be encoded as integers, or floating-point values, for use as inputs in machine learning 
algorithms. This process is called feature extraction (or vectorization). After this, we are about to calculate the IDF values by calling tfidf_transformer.fit on the word counts
we computed earlier. Now we will import Multinomialnb. This typically requires integer feature counts but tfidf may also work in practicality" which we have found out before this.
Basically, here we are implementing Naïve Bayes to predict if the messages are spam or ham. At last, we run the model for test size=0.2, and then through pipelining we predict spam 
or ham for the whole model. "Pipelines have functionalities namely fit/transform/predict, by this we can fit the entire pipeline to the training data and change to the test data 
without doing it independently.

Before running, please import pandas, numpy, seaborn, matplotlib, nltk or run the project on google collab. 
