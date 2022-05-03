#Script to get the total number of words, number of comments, and average number of words per comment
import csv
import os
import re
from gensim.models import Word2Vec
from tqdm import tqdm

number_of_comments = 0
all_words = 0

for file in tqdm(os.scandir("FileName")):
    # Do for all files in the folder
    with open(file) as file_obj:
        # ignore the heading
        to_ignore_heading = next(file_obj)
        reader = csv.reader(file_obj)
        for row in reader:
            # Clean comments before adding them to the list
            comment_to_be_added = re.sub('[^a-zA-Z0-9]', " ", row[0])
            all_words += len(comment_to_be_added.split())
            number_of_comments += 1

print(all_words, number_of_comments, all_words//number_of_comments)




