import cv2
import pafy
import numpy as np
from imread_from_url import imread_from_url

from mobileHumanPose import MobileHumanPose, YoloV5s
from mobileHumanPose.utils_pose_estimation import draw_skeleton, draw_heatmap, vis_3d_multiple_skeleton

draw_detections = False
draw_3dpose = True # TODO: make 3d plot faster

# Camera parameters for the deprojection
# TODO: Correct the deprojection function to properly transform the joints to 3D
focal_length = [None, None]
principal_points = [None, None]

pose_model_path='models/mobile_human_pose_working_well_256x256.onnx'
pose_estimator = MobileHumanPose(pose_model_path, focal_length, principal_points)

# Initialize person detector
detector_model_path='models/model_float32.onnx' 
person_detector = YoloV5s(detector_model_path, conf_thres=0.45, iou_thres=0.4)

# Initialize video
# cap = cv2.VideoCapture("video.mp4")

videoUrl = 'https://youtu.be/SJ6f2TnHZBc'
videoPafy = pafy.new(videoUrl)
print(videoPafy.streams)
cap = cv2.VideoCapture(videoPafy.streams[-1].url)
cap.set(cv2.CAP_PROP_POS_MSEC, 1*60000+30000) # Skip inital frames

out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (3840,720))

cv2.namedWindow("Estimated pose", cv2.WINDOW_NORMAL)

skip_detection = False
while cap.isOpened():
    try:
        # Read frame from the video
        ret, frame = cap.read()
    except:
        continue

    if ret: 

        # Detect people in the frame, but skip once every frame to increase speed
        if not skip_detection:
            boxes, detection_scores = person_detector(frame) 
        skip_detection = not skip_detection

        # Skip pose estimation if no person has been detected
        if boxes is not None:

            # Simulate depth based on the bouding box area
            areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
            depths = 500/(areas/(frame.shape[0]*frame.shape[1]))+500

            pose_img = frame.copy()
            if draw_detections:
                pose_img = person_detector.draw_detections(pose_img, boxes, detection_scores)

            heatmap_viz_img = frame.copy()
            img_heatmap = np.empty(frame.shape[:2])
            pose_3d_list = []
            for i, bbox in enumerate(boxes):
                
                keypoints, pose_3d, person_heatmap, scores = pose_estimator(frame, bbox, depths[i])
                pose_img = draw_skeleton(pose_img, keypoints, bbox[:2], scores)

                # Add the person heatmap to the image heatmap
                img_heatmap[bbox[1]:bbox[3],bbox[0]:bbox[2]] += person_heatmap 

                pose_3d_list.append(pose_3d)

            # Draw heatmap
            heatmap_viz_img = draw_heatmap(heatmap_viz_img, img_heatmap)

            # Draw 3D pose
            if draw_3dpose:
                vis_kps = np.array(pose_3d_list)
                img_3dpos = vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps))
                img_3dpos = cv2.resize(img_3dpos[200:-200,150:-150], frame.shape[1::-1])

                combined_img = np.hstack((heatmap_viz_img, pose_img, img_3dpos))
            else:
                combined_img = np.hstack((heatmap_viz_img, pose_img))

            out.write(combined_img)
            cv2.imshow("Estimated pose", combined_img)

        else:
            if draw_3dpose:
                combined_img = np.hstack((frame, frame, frame))
            else:
                combined_img = np.hstack((frame, frame))

            print("No person was detected")
            cv2.imshow("Estimated pose", combined_img)

    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

out.release()