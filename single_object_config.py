# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os
from sys import platform
from IPython import get_ipython

# base path directory
if platform == "win32":
    path_dir = os.path.dirname(__file__)
    
elif 'google.colab' in str(get_ipython()):
    path_dir = "/content/object-tracking-dlib"

# define the directory path to the input videos
input_dir = os.path.join(path_dir, "input")

# define the directory path to the output videos
output_dir = os.path.join(path_dir, "output")

# define the directory path to the mobilenet model
cnn_caffe_dir = os.path.join(path_dir, "mobilenet_ssd")

# define the directory path to the yolo-coco model
cnn_yolo_dir = os.path.join(path_dir, "yolo-coco")

# define the class label which is of interest
label = "person"

# define threshold confidence to filter weak detections
thres_confidence = 0.2
yolo_thres_confidence = 0.5
