import cv2
import sys
import os

from multiprocessing.pool import ThreadPool as Pool

### usage
### this tool will recursively convert all avis to directories with all frames as jpgs
###
###
### python3 extract_jpgs_from_avis.py <path_to_all_the_avis_of_the_dataset>
### python3 extract_jpgs_from_avis.py /home/jzf/Tasks__Gestures_Classification/3DCDC-NAS/Dataset/test

def get_unique_directories(paths):
    unique_dirs = set()
    for path in paths:
        directory = os.path.dirname(path)
        unique_dirs.add(directory)
    return list(unique_dirs)

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

cpuCount = os.cpu_count()
pool_size = int(0.9*cpuCount)
print("using %s CPUS!" % pool_size)

errors = []

class avi2jpg:

    def process(videopath):
        print(' + processing new video: %s' % (videopath))
        try:
            vidcap = cv2.VideoCapture(videopath)
            success,image = vidcap.read()
            count = 0
            name = videopath.split(".")[0]
            try:
                os.mkdir(name)
            except OSError as e:
                pass
            while success:
                framecount = "{number:06}".format(number=count)
                cv2.imwrite(os.path.join(name, framecount+".jpg"), image)   
                success,image = vidcap.read()
                count += 1
        except Exception as e:
            errors.append((videopath, e))
        else:
            print(' - %s successfully processed: %s' % (name, count))

    def convert(self, path):
        # Find all .avi files in the directory
        videos = find_avi_files(path)
        pool = Pool(pool_size)
        for video in videos:
            pool.apply_async(avi2jpg.process, (video,))
        pool.close()
        pool.join()
        print("============================================")
        unique_directories = get_unique_directories(videos)
        for directory in unique_directories:
            print(directory)
        print("============================================")
        for video, error in errors:
            print("------------------")
            print(video)
            print(error)
            print("------------------")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please give a video as an argument to this program")
    else:
        converter = avi2jpg()
        converter.convert(sys.argv[1])
