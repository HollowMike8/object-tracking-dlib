# object-tracking
 
Project to use a deep learning algorithm for object detection and implement a tracking algorithm to track the object

Note: Only single object detection is performed as a PoC

Combinations of two object detection alogirithms and two tracking algorithms have been used to compare the results

Object Detection : MobileNetSSD vs YOLOv3 <br />
Tracking         : dlib correlation tracker vs kalman filter

### Trials performed:
1. MobileNetSSD +  dlib correlation tracker
2. YOLOv3 + dlib correlation tracker
3. YOLOv3 + dlib correlation tracker vs kalman filter

### Results: 
Object Detection : Used single detection with the largest confidence

#### MobileNetSSD +  dlib correlation tracker:
1. Object detection is performed once in every 60 frames
2. Object detections (except initial) use additional criteria of checking if the new detection is close to previous bounding box (from tracking)
3. mobilenet_ssd fails in some detection steps (for refresh_rate = 30 frames) due to occlusion
4. Unsuccessful detection steps are skipped and tracking is used as before
5. Tracking is re-initiated but with the last successful bounding box
