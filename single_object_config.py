# -----------------------------
#   IMPORTS
# -----------------------------
# Import the necessary packages
import os

# base path directory
path_dir = "/content/object-tracking-dlib"

# define the directory path to the input videos
input_dir = os.path.join(path_dir, "input")

# define the directory path to the output videos
output_dir = os.path.join(path_dir, "output")

# define the directory path to the mobilenet model
cnn_caffe_dir = os.path.join(path_dir, "mobilenet_ssd")

# define the class label which is of interest
label = "person"

# define threshold confidence to filter weak detections
thres_confidence = 0.2
yolo_thres_confidence = 0.5
