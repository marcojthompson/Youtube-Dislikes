# Credit: https://github.com/Raka-Raka/download-youtube-comments-with-python/blob/main/download-youtube-comments-with-python.py

# Scrape Or Download Comments Using Python Through The Youtube Data API
# Watch the youtube video for explaination
# https://youtu.be/B9uCX2s7y7A
import Playlist_Vid_Id_Getter
import pandas as pd
import re
from string import ascii_letters
from apiclient.discovery import build

api_key = "API-KEY" # Replace this dummy api key with your own.

youtube = build('youtube', 'v3', developerKey=api_key)

def scrape_comments_with_replies(ID):
    box = [['Comment', 'Likes', 'Reply Count']]

    data = youtube.commentThreads().list(part='snippet', videoId=ID, maxResults='100', textFormat="plainText").execute()

    for i in data["items"]:
        #to limit the number of top level comments collected per video to 1000
        if len(box) > 1001:
            break
        comment = re.sub('[^a-zA-Z0-9]', " ", i["snippet"]['topLevelComment']["snippet"]["textDisplay"])
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
                comment = re.sub('[^a-zA-Z0-9]', " ", i["snippet"]["textDisplay"])
                likes = i["snippet"]['likeCount']
                replies = ""

                #Modified line to discard irrelevent comments
                if likes > 0:
                    box.append([comment, likes, replies])

    while ("nextPageToken" in data):

        data = youtube.commentThreads().list(part='snippet', videoId=ID, pageToken=data["nextPageToken"],
                                             maxResults='100', textFormat="plainText").execute()

        for i in data["items"]:
            #to limit the number of top level comments collected per video to 1000
            if len(box) > 1001:
                break
            comment = re.sub('[^a-zA-Z0-9]', " ", i["snippet"]['topLevelComment']["snippet"]["textDisplay"])
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
                    comment = re.sub('[^a-zA-Z0-9]', " ", i["snippet"]["textDisplay"])
                    likes = i["snippet"]['likeCount']
                    replies = ''

                    #Modified line to discard irrelevent comments
                    if likes > 0:
                        box.append([comment, likes, replies])

    df = pd.DataFrame({'Comment': [i[0] for i in box], 'likes': [i[1] for i in box], 'Reply Count': [i[2] for i in box]})

    df.to_csv(path_or_buf="FileName/" + str(ID) + '.csv', index=False, header=False)

    return "Success!"


VideoIds = Playlist_Vid_Id_Getter.getVideoIdsFromPlaylist("Playlist-Link")


for i in VideoIds:
    scrape_comments_with_replies(i)