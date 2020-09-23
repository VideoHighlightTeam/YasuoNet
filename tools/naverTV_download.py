import youtube_dl
import sys
from requests import get

class Downloader:
    def __init__(self, resolution):
        self.resolution = resolution
        
    def get_youtube_url(self, url):
        result = ""
        ydl_opts = {"geo_bypass_country":"US"}
        # create youtube-dl object
        ydl = youtube_dl.YoutubeDL(ydl_opts)
        # set video url, extract video information
        info_dict = ydl.extract_info(url, download=False)
        # get video formats available
        formats = info_dict.get('formats',None)
        for f in formats:
            # I want the lowest resolution, so I set resolution as 144p
            if f.get('format_id',None) == 'avc1_' + self.resolution + 'P': #270, 370 ,...
                #get the video url
                result = f.get('url',None)
        return result
    
    def download(self, url, file_name):
        dl_url = self.get_youtube_url(url)
        with open(file_name + '.mp4', "wb") as file:   
                response = get(dl_url)               
                file.write(response.content)

def main():
    if len(sys.argv) < 2:
        print('인자를 입력해주세요.')
        return
    
    file_name = sys.argv[1]
    resolution = sys.argv[2]
    
    dl = Downloader(resolution)
    with open(file_name) as f:
        while True:
            line = f.readline()
            if not line:
                break
            arg = line.split(',')
            dl.download(arg[0], arg[1].strip())

if __name__ == "__main__":
    main()
