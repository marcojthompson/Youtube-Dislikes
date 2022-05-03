# Credit: https://python.tutorialink.com/extract-individual-links-from-a-single-youtube-playlist-link-using-python/

import googleapiclient.discovery
from urllib.parse import parse_qs, urlparse

# extract playlist id from url
def getVideoIdsFromPlaylist(url):
    query = parse_qs(urlparse(url).query, keep_blank_values=True)
    playlist_id = query["list"][0]
    print(f'get all playlist items links from {playlist_id}')
    youtube = googleapiclient.discovery.build("youtube", "v3", developerKey = "API-KEY")

    request = youtube.playlistItems().list(
        part = "snippet",
        playlistId = playlist_id,
        maxResults = 50
    )
    response = request.execute()

    playlist_items = []
    while request is not None:
        response = request.execute()
        playlist_items += response["items"]
        request = youtube.playlistItems().list_next(request, response)

    videoIds = [i['snippet']['resourceId']['videoId'] for i in playlist_items]

    return videoIds


