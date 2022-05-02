#Credit: https://machinelearningmastery.com/develop-word-embeddings-python-gensim/
import csv
import os
import re
from gensim.models import Word2Vec
from tqdm import tqdm

list_of_comments = []

for file in tqdm(os.scandir("Comments-Test")):
    #Do for all files in the folder
    with open(file) as file_obj:
        #ignore the heading
        to_ignore_heading = next(file_obj)
        reader = csv.reader(file_obj)
        for row in reader:
            #Clean comments before adding them to the list
            comment_to_be_added = re.sub('[^a-zA-Z0-9]', " ", row[0])
            list_of_comments.append(comment_to_be_added)

#print(list_of_comments)

model = Word2Vec(list_of_comments, workers = 2)
print(model.wv["moment"])




