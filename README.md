# ONNX-Mobile-Human-Pose-3D
Python scripts for performing 3D human pose estimation using the Mobile Human Pose model.

![Mobile Human 3D Pose mation ONNX](https://github.com/ibaiGorordo/ONNX-Mobile-Human-Pose-3D/blob/main/doc/img/output.bmp)
*Source: (https://static2.diariovasco.com/www/pre2017/multimedia/noticias/201412/01/media/DF0N5391.jpg)*

### :exclamation::warning: Known issues

 * The models works well when the person is looking forward and without occlusions, it will start to fail as soon as the person is occluded.
 * The model is fast, but the 3D representation is slow due to matplotlib, this will be fixed. The 3d representation can be ommitted for faster inference by setting **draw_3dpose** to False

# Requirements

 * **OpenCV**, **imread-from-url**, **scipy**, **onnx** and **onnxruntime**. Also, **pafy** and **youtube-dl** are required for youtube video inference. 
 
# Installation
```
pip install -r requirements.txt
pip install pafy youtube-dl
```

# ONNX model
The original models were converted to different formats (including .onnx) by [PINTO0309](https://github.com/PINTO0309), download the models from [his repository](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/135_CoEx) and save them into the **[models](https://github.com/ibaiGorordo/ONNX-Mobile-Human-Pose-3D/tree/main/models)** folder. 

 * YOLOv5s: You will also need an object detector to first detect the people in the image. Download alos the model from the [model zoo] (https://github.com/PINTO0309/PINTO_model_zoo/blob/main/059_yolov5/22_yolov5s_new/download.sh) and save the .onnx version into the **[models](https://github.com/ibaiGorordo/ONNX-Mobile-Human-Pose-3D/tree/main/models)** folder.

# Original model
The original model was taken from the [original repository](https://github.com/SangbumChoi/MobileHumanPose).
 
# Examples

 * **Image inference**:
 
 ```
 python imagePoseEstimation.py 
 ```
 
  * **Video inference**:
 
 ```
 python videoPoseEstimation.py
 ```
 
 * **Webcam inference**:
 
 ```
 python webcamPoseEstimation.py
 ```
 
# [Inference video Example](https://youtu.be/bgjKKbGp5uo) 
 ![Mobile Human 3D Pose mation ONNX](https://github.com/ibaiGorordo/ONNX-Mobile-Human-Pose-3D/blob/main/doc/img/Mobile%20Pose%20Estimation%20ONNX.gif)

# References:
* Mobile human pose model: https://github.com/SangbumChoi/MobileHumanPose
* PINTO0309's model zoo: https://github.com/PINTO0309/PINTO_model_zoo
* PINTO0309's model conversion tool: https://github.com/PINTO0309/openvino2tensorflow
* 3DMPPE_POSENET_RELEASE repository: https://github.com/mks0601/3DMPPE_POSENET_RELEASE
* Original YOLOv5 repository: https://github.com/ultralytics/yolov5
* Original paper: 
https://openaccess.thecvf.com/content/CVPR2021W/MAI/html/Choi_MobileHumanPose_Toward_Real-Time_3D_Human_Pose_Estimation_in_Mobile_Devices_CVPRW_2021_paper.html
 


