import cv2
import sys
import os

### usage
### this tool will recursively convert all avis to directories with all frames as jpgs
###
###
### python3 extract_jpgs_from_avis.py <path_to_all_the_avis_of_the_dataset>
###

def find_avi_files(directory):
    avi_files = []
    # Walk through the directory tree recursively
    for root, dirs, files in os.walk(directory):
        # Check each file in the current directory
        for file in files:
            # Check if the file has a .avi extension
            if file.endswith(".avi"):
                # If so, add its full path to the list
                avi_files.append(os.path.join(root, file))
    return avi_files

class avi2jpg:
    def convert(self, path):
        # Find all .avi files in the directory
        videos = find_avi_files(path)
        for video in videos:
            vidcap = cv2.VideoCapture(video)
            success,image = vidcap.read()
            count = 0
            name = video.split(".")[0]
            try:
                os.mkdir(name)
            except OSError as e:
                print(e)
            while success:
                framecount = "{number:06}".format(number=count)
                cv2.imwrite(os.path.join(name, framecount+".jpg"), image)     # save frame as JPEG file      
                success,image = vidcap.read()
                count += 1
            print('%s read a new video: %s' % (name, count))

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a video as an argument to this program")
    else:
        converter = avi2jpg()
        converter.convert(sys.argv[1])
