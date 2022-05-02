import requests as rq
import json
import os
import time
import numpy as np
import pandas as pd 
from tqdm import tqdm
import Playlist_Vid_Id_Getter

id_list =[]

#retrieves like count, dislike count, and view count for every video from the YouTube dislike extension and returns a json
def retrieve_data(id):
        url = "https://returnyoutubedislikeapi.com/votes?videoId=%s" % (id)
        data = rq.get(url).json()
        return data

# IDs of videos(can be used if you have acccess to the YouTube api)
# id_list = sorted(Playlist_Vid_Id_Getter.getVideoIdsFromPlaylist("https://www.youtube.com/playlist?list=PLWF14W8Rw2HBB9NBJ-avYD7fMlbvGdpz4"))

# IDs of videos(use if you have all the video ids in some other way)
for file in tqdm(os.scandir("Comments-TechLead")):
    f = open(file, "r")
    id_list.append(f.name.split("/")[1][0:-4])

#retrieves data from all vidoes and loads into data_array 
data_array = []
for id in id_list:
    data = retrieve_data(id) #returns json data for the id 
    a = (data["id"],data["likes"], data["dislikes"],data["viewCount"])
    data_array.append(a)
    time.sleep(1.0) #The site can only handle one request per second

#convert to np array then to pd dataframe 
numpy_array = np.array(data_array)
df = pd.DataFrame(numpy_array, columns = ['ID','Likes','Dislikes','View Count'])

#create csv file
df.to_csv('dislike_data.csv')
