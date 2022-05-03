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


#PART 2: TF-IDF
#method to get all comments in a single string for a given video for TF-IDF:
def get_comments_tfidf(vid_id):
    comm_list=''
    df = pd.read_csv(path+"/Comments-TechLead/"+vid_id)
    for index, row in df.iterrows():
        comm_list = comm_list+ df.loc[index, 'Comment']+' ' 
    return comm_list

#get all comments for TF-IDF: in this list, each item is a single string with ALL the comments for a video
all_comments=[]
for index, row in likes_df.iterrows():
    all_comments.append(get_comments_tfidf(likes_df.loc[index, 'ID']+'.csv'))

#preprocess the comments: get rid of stop words, lemmatize words, remove symbols
def preprocess(data):
    data=data.split()
    data = np.char.lower(data) #convert to lowercase

    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n," #remove punctuation
    for i in symbols:
        data = np.char.replace(data, i, ' ')

    stop_words=stopwords.words('english')

    lemmatizer = WordNetLemmatizer() #converts words to their root form
    ps = PorterStemmer()
    data2=''
    for word in data:
        w = ps.stem(lemmatizer.lemmatize(word))
        if word not in stop_words:
            data2 = data2+w+' '
    data=data2

    return data

#actually preprocess the data
all_comments2 = []
for i in all_comments:
    all_comments2.append(preprocess(i))

#Get the document frequency
DF = {}
for i in range(len(all_comments2)):
    tokens = all_comments2[i]
    for word in tokens.split():
        try:
            DF[word].add(i)
        except:
            DF[word] = {i}

#we don't actually need the list of documents a word appears in, just HOW MANY documents
for i in DF:
    DF[i] = len(DF[i])
    
total_vocab = [x for x in DF]

#find TF-IDF
tf_idf = pd.DataFrame()
for i in range(len(all_comments2)):
    tokens = all_comments2[i]
    counter = Counter(tokens.split())
    for token in np.unique(tokens.split()):
        tf = counter[token]/len(tokens.split())
        df=DF[token]
        idf = np.log(len(all_comments2)/df)
        tf_idf.loc[i, token] = tf*idf
tf_idf=tf_idf.fillna(0)
display(tf_idf)


#PART 3: Decision Tree

#add target column to the tf_idf matrix
for index, row in tf_idf.iterrows():
    tf_idf.loc[index, 'ratio_likesdislikes'] = likes_df.loc[index, 'Likes']/(likes_df.loc[index, 'Likes']+likes_df.loc[index, 'Dislikes'])
tf_idf.head()

#feature selection
X_ta = tf_idf.iloc[:, :-1].to_numpy()
y_ta = tf_idf.iloc[:, -1].to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X_ta, y_ta)
print(len(y_test))

# Create Decision Tree classifer object
clf = DecisionTreeRegressor()

# Train the classifier
tree= clf.fit(X_train,y_train)

p=clf.predict(X_test)

#plot predicted and actual like ratio together for each video to observe trends.
plt.plot(p, label='Predicted Ratio')
plt.plot(y_test, label='Actual Ratio')
plt.legend()
plt.ylim(0,1)
plt.xlabel("Video")
plt.ylabel("Fraction of Likes")
plt.title("Like Ratio: Predicted vs Actual")
plt.show()

#plot predicted vs actual like ratio
plt.scatter(y_test,p)
m, b = np.polyfit(p, y_test, 1)
plt.plot(p, m*p + b, c='g', label="Line of Best Fit")
plt.xlabel("Actual fraction")
plt.ylabel("Predicted Fraction")
plt.title("Like Ratio: Predicted vs Actual")
min_val = np.min([np.min(p), np.min(y_test)])
plt.legend()
plt.xlim(min_val-0.05,1)
plt.ylim(min_val-0.05,1)
plt.show()

#corr coeff
print(np.corrcoef(y_test,p)[0,1])

#PART 6: Word Clouds
from wordcloud import WordCloud
import  matplotlib.pylab as plt

# from all comments
comments_string= " ".join(all_comments2)
wc_comments = WordCloud(max_font_size=40).generate(comments_string)

plt.figure(figsize=(20, 8))

plt.axis("off")
plt.imshow(wc_comments, interpolation="bilinear")

#now, iterate over each file IN the folder
def vid_comments(id):
    comm_list=''
    df = pd.read_csv(path+"/Comments-TechLead/"+id)
    for index, row in df.iterrows():
        comm_list = comm_list+ df.loc[index, 'Comment']+' '
    return comm_list

#highest ratio comment wordcloud
highest_ratio = vid_comments("1PSIdXMMn7I.csv")

wc_highest = WordCloud(max_font_size=40).generate(highest_ratio)
plt.figure(figsize=(20, 8))
plt.subplot(1, 3, 1)
plt.title("highest like ratio")
plt.axis("off")
plt.imshow(wc_highest, interpolation="bilinear")

#lowest ratio comment wordcloud
lowest_ratio = vid_comments("IEmpYqIAivw.csv")

wc_lowest = WordCloud(max_font_size=40).generate(lowest_ratio)
plt.figure(figsize=(20, 8))
plt.subplot(1, 3, 1)
plt.title("lowest like ratio")
plt.axis("off")
plt.imshow(wc_lowest, interpolation="bilinear")

df = pd.read_csv(path+"/"+'dislike_data.csv')
ratio_list =[]
view_list = []
for index, row in df.iterrows():
    df.loc[index, 'Ratio'] = df.loc[index, 'Likes'] / (df.loc[index, 'Likes']+df.loc[index, 'Dislikes'])
    ratio_list.append( df.loc[index, 'Ratio'])
    view_list.append( df.loc[index, 'View Count'])

plt.scatter(ratio_list, view_list)
plt.title("Like Ratio vs. View Count")
plt.xlabel("like ratio")
plt.ylabel("view count")
plt.show()