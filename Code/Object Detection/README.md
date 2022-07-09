# Object Detection


Besides roads, buildings can also be considered an effective way to georeference an
aerial image, taken from a UAV, with a referenced map. This work focuses on road
detection; however, roads and buildings may cooperate and produce better results. The
YOLO family of networks have been proven to be very accurate while dealing with
features in aerial images, and thus, these methods have been chosen for this work’s
system.

## YOLO - You Only Look Once
You Only Look Once (YOLO) created by Joseph Redmon, Santosh Divvala, Ross
Girshick, and Ali Farhadi (Redmon et al., 2015) is a state-of-the-art, real-time object
detection system.

The below image is a sample of YOLO system, where the image is divided into an S*S grid
of cells. Then by creating bounding boxes and calculating the confidence the system
predict the objects inside the image

![YoloSystem](https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Code/Object%20Detection/Images/YoloSystem.png?raw=true)

### YOLOv3
YOLOv3 (Redmon and Farhadi, 2018) is the latest upgrade of YOLO, in which the
speed has been traded off for more accuracy, as the earlier 45 FPS has been decreased to
30 FPS.However, the accuracy has been rapidly increased due to the complexity of the
new architecture called Darknet53, which uses 53 convolutional layers plus additional
object detection layers that total to 106 layers.

Thus, Darknet53 is the backbone of YOLOv3, and it is the reason for extracting features and directly relate
them to the relative objects in the image. The architecture can be shown here:
![YOLOv3](https://github.com/TypEktor/Aerial-Image-Geolocalisation-Using-Road-Detection/blob/main/Code/Object%20Detection/Images/Yolov3.png?raw=true)
