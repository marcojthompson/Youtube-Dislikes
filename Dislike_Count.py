import requests as rq
import json
import numpy as np
import pandas as pd 
import Playlist_Vid_Id_Getter

#retrieves like count, dislike count, and view count for every video from the YouTube dislike extension and returns a json
def retrieve_data(id):
        url = "https://returnyoutubedislikeapi.com/votes?videoId=%s" % (id)
        data = rq.get(url).json()
        return data

# IDs of videos
id_list = sorted(Playlist_Vid_Id_Getter.getVideoIdsFromPlaylist("https://www.youtube.com/playlist?list=PLWF14W8Rw2HBB9NBJ-avYD7fMlbvGdpz4"))

#retrieves data from all vidoes and loads into data_array 
data_array = []
for id in id_list:
    data = retrieve_data(id) #returns json data for the id 
    a= (data["id"],data["likes"], data["dislikes"],data["viewCount"])
    data_array.append(a)

#convert to np array then to pd dataframe 
numpy_array = np.array(data_array)
df = pd.DataFrame(numpy_array, columns = ['ID','Likes','Dislikes','View Count'])

#create csv file
df.to_csv('dislike_data.csv')