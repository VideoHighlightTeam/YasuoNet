import cv2
import sys

class Video_capture:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.ret = None 
        self.frame = None 
        
    def move_frame(self, frame_cnt):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_cnt)
        self.read()
        
    def make_grayscale(self):
        return cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)
    
    def read(self):
        self.ret, self.frame = self.cap.read()
        return

class Highlight_finder:
    def __init__(self, source, highlight, precision):
        self.name = source[:-4]
        self.source = Video_capture(source)
        self.highlight= Video_capture(highlight)
        self.frame_cnt = 0
        self.highlight_frame_cnt = 210 # 네이버 로고 지난 프레임
        self.precision = precision
        self.result = []
    
    def load_video(self):
        if not self.source.cap.isOpened():
                print('source video not opened')
                exit(-1)
        if not self.highlight.cap.isOpened():
                print('highlight video not opened')
                exit(-1)
        self.source.move_frame(self.frame_cnt)
        self.highlight.move_frame(self.highlight_frame_cnt)       
        return
    
    def match_template(self, src, frame):
        R = cv2.matchTemplate(src, frame, cv2.TM_CCORR_NORMED)
        _, maxVal, _, maxLoc = cv2.minMaxLoc(R)
        return maxVal
    
    def find_highlight(self):
        start = 0
        end = 0
        f = open(self.name +".txt",'w')
        while True: # 영상이 끝날때까지
            self.frame_cnt += 5
            if self.frame_cnt % 1800 == 0:
                print(self.frame_cnt / 1800, "min", self.highlight_frame_cnt)
            for i in range(5):
                self.source.read()
            if not self.source.ret or not self.highlight.ret:
                break
        #---------------------------------------------------------------------
            maxVal = self.match_template(self.source.make_grayscale(), self.highlight.make_grayscale())

            #cv2.imshow('source', self.source.frame)
            key = cv2.waitKey(25)
            if key == 27:
                cv2.destroyAllWindows()
                break

            if maxVal < self.precision: # 영상이 일치하는지 체크
                continue
            else: # 일치한다면
                result = maxVal
                if(self.frame_cnt - end > 150):
                    if end != 0 and end - start >= 60:
                        start_time = '%d:%d:%d' % (start // 29.97 // 3600, start//29.97%3600//60, start//29.97%60)
                        end_time = '%d:%d:%d' % (end // 29.97 // 3600, end//29.97%3600//60 , end//29.97%60)
                        f.write(str(start) + "," + str(end)+", " + start_time + "," + end_time+"\n")
                    start = self.frame_cnt
                while True: # 일치하지 않는 구간까지 반복
                    print(self.frame_cnt, result, self.highlight_frame_cnt)
                    self.highlight_frame_cnt += 15
                    self.frame_cnt += 15
                    for i in range(15):
                        self.highlight.read()
                        self.source.read()
                        
                    '''
                    cv2.imshow('source', self.source.frame)
                    cv2.imshow('highlight', self.highlight.frame)
                    key = cv2.waitKey(25)
                    if key == 27:
                        cv2.destroyAllWindows()
                    '''
                    if not self.source.ret or not self.highlight.ret: 
                        break
                    result = self.match_template(self.source.make_grayscale(), self.highlight.make_grayscale())
                    if result > self.precision: # 일치한다면
                        end = self.frame_cnt
                        continue
                    else: # 일치하지 않는다면
                        print(self.frame_cnt, result, self.highlight_frame_cnt)
                        end = self.frame_cnt - 15
                        self.highlight_frame_cnt += 30
                        for i in range(30):
                            self.highlight.read()
                        break
                
        #----------------------------------------------------------------------
        if self.source.cap.isOpened():
            self.source.cap.release()

        if self.highlight.cap.isOpened():
            self.highlight.cap.release()

        #cv2.destroyAllWindows()
        print("END")
        return

def main():
    if len(sys.argv) < 1:
        print('인자를 입력해주세요.')
        return
    
    text_path = sys.argv[1]
    try:
        precision = int(sys.argv[2])
    except:
        precision = 0.95
    
    
    with open(text_path, "r") as src:
        while(True):
            source_path = src.readline()
            if not source_path : break
            source_path = "./data/" + source_path.split(',')[1].strip()
            highlight_path = source_path + "_HL.mp4"
            source_path += ".mp4"
            print(source_path)
            finder = Highlight_finder(source_path, highlight_path, precision)
            finder.load_video()
            finder.find_highlight()

if __name__ == "__main__":
    main()
