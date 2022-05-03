#NOTE: This code is meant to be executed in a Jupyter Notebook 
#within a virtual environment. Please check to make sure all 
#required packages have been installed before running.


#PART 1: Importing Data

#Package installation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
from collections import Counter
from sklearn.tree import DecisionTreeRegressor
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
from scipy.optimize import curve_fit
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

#first, take in the likes, dislikes csv
path=r'C:\Users\krazy\Documents\BigData\TechLead'
#path = f"/Users/emilyalameddine/Desktop/School/project"
likes_df = pd.read_csv(path+"/"+'dislike_data.csv')
display(likes_df)


#PART 4: Vader

#method to get list of comments for a given video in a form for VADER: each list item is a different comment
def get_comments(vid_id):
    df = pd.read_csv(path+"/Comments-TechLead/"+vid_id)
    return df.loc[:,'Comment']

#method to get sentiment scores for a set of comments
def comments_sentiment(comments_list):    
    model = SentimentIntensityAnalyzer()
    pos_scores=0
    neg_scores=0
    for comment in comments_list:
        scores = model.polarity_scores(comment)
        pos_scores += scores['pos']
        neg_scores += scores['neg']
    num_comments = len(comments_list)
    pos = pos_scores/num_comments
    neg = neg_scores/num_comments
    return pos, neg

#PART 5: Regression based on sentiment scores

#first, take in the likes, dislikes csv to find the ratio for each video
#also, add in sentiment numbers
likes_df = pd.read_csv(path+"/"+'dislike_data.csv')
for index, row in likes_df.iterrows():
    likes_df.loc[index, 'Ratio'] = likes_df.loc[index, 'Likes'] / (likes_df.loc[index, 'Likes']+likes_df.loc[index, 'Dislikes'])
    pos, neg = comments_sentiment(get_comments(likes_df.loc[index, 'ID']+'.csv'))
    likes_df.loc[index, 'Pos Sentiment'] = pos
    likes_df.loc[index, 'Neg Sentiment'] = neg
display(likes_df)

#pos vs neg sentiment graph
pos=likes_df.loc[:, 'Pos Sentiment']
neg=likes_df.loc[:, 'Neg Sentiment']
plt.scatter(pos, neg)
plt.xlabel("Positive Sentiment")
plt.ylabel("Negative Sentiment")
plt.title("Positive vs. Negative Sentiment in Comments")
plt.xlim(np.min(pos)-0.01,np.max(pos)+0.01)
plt.ylim(np.min(neg)-0.01,np.max(neg)+0.01)
plt.show()

#corr coeff for sentiment
print(np.corrcoef(pos, neg)[0,1])

def logistic_regression(x, w0, w1, w2):
    f=x[0] * w0 + x[1] * w1 + w2
    return 1-(1 / (1 + np.exp(f)))

#define x and y (training set and testing set)
X_ta = likes_df.iloc[:, 6:8].to_numpy()
y_ta = likes_df.iloc[:, 5].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_ta, y_ta)

optimal_params, cov_params = curve_fit(logistic_regression, X_train.T, y_train)
optimal_params

#make predictions on X_test
pred=[]
for i in X_test:
    pred.append(logistic_regression(i, *optimal_params))

#plot predicted and actual like ratio together for each video to observe trends.
plt.plot(pred, label='Predicted Ratio')
plt.plot(y_test, label='Actual Ratio')
plt.legend()
plt.ylim(0,1)
plt.xlabel("Video")
plt.ylabel("Fraction of Likes")
plt.title("Like Ratio Predicted vs Actual")
plt.show()

#plot predicted vs actual like ratio
plt.scatter(y_test, pred)
m, b = np.polyfit(p, y_test, 1)
plt.plot(p, m*p + b, c='g')
plt.xlabel("Actual fraction")
plt.ylabel("Predicted Fraction")
plt.title("Like Ratio: Predicted vs Actual")
min_val = np.min([np.min(pred), np.min(y_test)])
plt.xlim(min_val-0.05,1)
plt.ylim(min_val-0.05,1)
plt.show()

#corr coeff
print(np.corrcoef(y_test,pred)[0,1])