# Credit: https://github.com/Raka-Raka/download-youtube-comments-with-python/blob/main/download-youtube-comments-with-python.py

# Scrape Or Download Comments Using Python Through The Youtube Data API
# Watch the youtube video for explaination
# https://youtu.be/B9uCX2s7y7A
import Playlist_Vid_Id_Getter
api_key = "AIzaSyBa89b5AH1zvxTFtBfCVZdJ0DRB8Sg05qg" # Replace this dummy api key with your own.

from apiclient.discovery import build
youtube = build('youtube', 'v3', developerKey=api_key)

import pandas as pd
import re
from string import ascii_letters

def scrape_comments_with_replies(ID):
    box = [['Comment', 'Likes', 'Reply Count']]

    data = youtube.commentThreads().list(part='snippet', videoId=ID, maxResults='100', textFormat="plainText").execute()

    for i in data["items"]:
        if len(box) > 1001:
            break
        #name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
        comment = re.sub('[^a-zA-Z0-9]', " ", i["snippet"]['topLevelComment']["snippet"]["textDisplay"])
        #published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
        likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
        replies = i["snippet"]['totalReplyCount']

        #Modified line to discard irrelevent comments
        if likes > 0:
            box.append([comment, likes, replies])

        totalReplyCount = i["snippet"]['totalReplyCount']

        if totalReplyCount > 0:

            parent = i["snippet"]['topLevelComment']["id"]

            data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                            textFormat="plainText").execute()

            for i in data2["items"]:
                #name = i["snippet"]["authorDisplayName"]
                comment = re.sub('[^a-zA-Z0-9]', " ", i["snippet"]["textDisplay"])
                #published_at = i["snippet"]['publishedAt']
                likes = i["snippet"]['likeCount']
                replies = ""

                #Modified line to discard irrelevent comments
                if likes > 0:
                    box.append([comment, likes, replies])

    while ("nextPageToken" in data):

        data = youtube.commentThreads().list(part='snippet', videoId=ID, pageToken=data["nextPageToken"],
                                             maxResults='100', textFormat="plainText").execute()

        for i in data["items"]:
            if len(box) > 1001:
                break
            #name = i["snippet"]['topLevelComment']["snippet"]["authorDisplayName"]
            comment = re.sub('[^a-zA-Z0-9]', " ", i["snippet"]['topLevelComment']["snippet"]["textDisplay"])
            #published_at = i["snippet"]['topLevelComment']["snippet"]['publishedAt']
            likes = i["snippet"]['topLevelComment']["snippet"]['likeCount']
            replies = i["snippet"]['totalReplyCount']

            #Modified line to discard irrelevent comments
            if likes > 0:
                box.append([comment, likes, replies])

            totalReplyCount = i["snippet"]['totalReplyCount']

            if totalReplyCount > 0:

                parent = i["snippet"]['topLevelComment']["id"]

                data2 = youtube.comments().list(part='snippet', maxResults='100', parentId=parent,
                                                textFormat="plainText").execute()

                for i in data2["items"]:
                    #name = i["snippet"]["authorDisplayName"]
                    comment = re.sub('[^a-zA-Z0-9]', " ", i["snippet"]["textDisplay"])
                    #published_at = i["snippet"]['publishedAt']
                    likes = i["snippet"]['likeCount']
                    replies = ''

                    #Modified line to discard irrelevent comments
                    if likes > 0:
                        box.append([comment, likes, replies])

    df = pd.DataFrame({'Comment': [i[0] for i in box], 'likes': [i[1] for i in box], 'Reply Count': [i[2] for i in box]})

    df.to_csv(path_or_buf="Comments-TechLead/" + str(ID) + '.csv', index=False, header=False)

    return "Successful! Check the CSV file that you have just created."


VideoIds = Playlist_Vid_Id_Getter.getVideoIdsFromPlaylist("https://www.youtube.com/playlist?list=PLWF14W8Rw2HBB9NBJ-avYD7fMlbvGdpz4")


#Run in increments
for i in VideoIds:
    scrape_comments_with_replies(i)
"""
for i in range(50, 100):
    scrape_comments_with_replies(VideoIds[i])

for i in range(100, 150):
    scrape_comments_with_replies(VideoIds[i])

for i in range(150, 200):
    scrape_comments_with_replies(VideoIds[i])

for i in range(200, 250):
    scrape_comments_with_replies(VideoIds[i])

for i in range(250, 300):
    scrape_comments_with_replies(VideoIds[i])

for i in range(300, len(VideoIds)):
    scrape_comments_with_replies(VideoIds[i])
"""