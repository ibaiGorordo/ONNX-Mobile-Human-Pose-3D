import cv2
import numpy as np
from imread_from_url import imread_from_url

from mobileHumanPose import MobileHumanPose, YoloV5s
from mobileHumanPose.utils_pose_estimation import draw_skeleton, draw_heatmap, vis_3d_multiple_skeleton

if __name__ == '__main__':
    
    draw_detections = False

    # Camera parameters for the deprojection
    # TODO: Correct the deprojection function to properly transform the joints to 3D
    focal_length = [None, None]
    principal_points = [None, None]

    pose_model_path='models/mobile_human_pose_working_well_256x256.onnx'
    pose_estimator = MobileHumanPose(pose_model_path, focal_length, principal_points)

    # Initialize person detector
    detector_model_path='models/model_float32.onnx' 
    person_detector = YoloV5s(detector_model_path, conf_thres=0.5, iou_thres=0.4)
 
    # image = cv2.imread("input.jpg")
    image = imread_from_url("https://static2.diariovasco.com/www/pre2017/multimedia/noticias/201412/01/media/DF0N5391.jpg")
    
    # Detect people in the image
    boxes, detection_scores = person_detector(image) 

    # Exit if no person has been detected
    if boxes is None:
        print("No person was detected")
        sys.exit()

    # Simulate depth based on the bouding box area
    areas = (boxes[:,2] - boxes[:,0]) * (boxes[:,3] - boxes[:,1])
    depths = 500/(areas/(image.shape[0]*image.shape[1]))+500

    # Draw detected person bounding boxes 
    pose_img = image.copy()
    if draw_detections:
        pose_img = person_detector.draw_detections(pose_img, boxes, detection_scores)

    # Initialize the represntation images 
    heatmap_viz_img = image.copy()
    img_heatmap = np.empty(image.shape[:2])
    pose_3d_list = []

    # Estimate the pose for each detected person
    for i, bbox in enumerate(boxes):
        
        # Draw the estimated pose
        keypoints, pose_3d, person_heatmap, scores = pose_estimator(image, bbox, depths[i])
        pose_img = draw_skeleton(pose_img, keypoints, bbox[:2], scores)

        # Add the person heatmap to the image heatmap
        img_heatmap[bbox[1]:bbox[3],bbox[0]:bbox[2]] += person_heatmap 

        # Add the 3d pose to the list
        pose_3d_list.append(pose_3d)

    # Draw heatmap
    heatmap_viz_img = draw_heatmap(heatmap_viz_img, img_heatmap)

    # Draw 3D pose
    vis_kps = np.array(pose_3d_list)
    img_3dpos = vis_3d_multiple_skeleton(vis_kps, np.ones_like(vis_kps))
    img_3dpos = cv2.resize(img_3dpos[200:-200,150:-150], image.shape[1::-1])

    # Combine the images for showing them together
    combined_img = np.hstack((heatmap_viz_img, pose_img, img_3dpos))

    cv2.imwrite("output.bmp", combined_img)
   
    cv2.namedWindow("Estimated pose", cv2.WINDOW_NORMAL)
    cv2.imshow("Estimated pose", combined_img)


    cv2.waitKey(0)