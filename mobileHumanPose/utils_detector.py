import numpy as np

# Source: https://github.com/PINTO0309/PINTO_model_zoo/blob/main/059_yolov5/22_yolov5s_new/yolov5_tflite_inference.py

def non_max_suppression(boxes, scores, threshold):	

    assert boxes.shape[0] == scores.shape[0]

    # bottom-left origin
    ys1 = boxes[:, 0]
    xs1 = boxes[:, 1]

    # top-right target
    ys2 = boxes[:, 2]
    xs2 = boxes[:, 3]

    # box coordinate ranges are inclusive-inclusive
    areas = (ys2 - ys1) * (xs2 - xs1)
    scores_indexes = scores.argsort().tolist()
    boxes_keep_index = []

    while len(scores_indexes):
        index = scores_indexes.pop()
        boxes_keep_index.append(index)
        if not len(scores_indexes):
            break
        ious = compute_iou(boxes[index], boxes[scores_indexes], areas[index],
                        areas[scores_indexes])
        filtered_indexes = set((ious > threshold).nonzero()[0])
        # if there are no more scores_index
        # then we should pop it
        scores_indexes = [
            v for (i, v) in enumerate(scores_indexes)
            if i not in filtered_indexes
        ]
    return np.array(boxes_keep_index)


def compute_iou(box, boxes, box_area, boxes_area):
    # this is the iou of the box against all other boxes
    assert boxes.shape[0] == boxes_area.shape[0]

    # get all the origin-ys
    # push up all the lower origin-xs, while keeping the higher origin-xs
    ys1 = np.maximum(box[0], boxes[:, 0])

    # get all the origin-xs
    # push right all the lower origin-xs, while keeping higher origin-xs
    xs1 = np.maximum(box[1], boxes[:, 1])

    # get all the target-ys
    # pull down all the higher target-ys, while keeping lower origin-ys
    ys2 = np.minimum(box[2], boxes[:, 2])

    # get all the target-xs
    # pull left all the higher target-xs, while keeping lower target-xs
    xs2 = np.minimum(box[3], boxes[:, 3])

    # each intersection area is calculated by the
    # pulled target-x minus the pushed origin-x
    # multiplying
    # pulled target-y minus the pushed origin-y
    # we ignore areas where the intersection side would be negative
    # this is done by using maxing the side length by 0
    intersections = np.maximum(ys2 - ys1, 0) * np.maximum(xs2 - xs1, 0)

    # each union is then the box area
    # added to each other box area minusing their intersection calculated above
    unions = box_area + boxes_area - intersections

    # element wise division
    # if the intersection is 0, then their ratio is 0
    ious = intersections / unions
    return ious

def xywh2xyxy(x, img_width, img_height):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    num_boxes= x.shape[0]
    zero_array = np.zeros(num_boxes, dtype=int)
    width_array = np.ones(num_boxes, dtype=int)*img_width
    height_array = np.ones(num_boxes, dtype=int)*img_height

    y = x.copy()
    y[:, 0] = x[:, 0] - x[:, 2] // 2 # top left x
    y[:, 1] = x[:, 1] - x[:, 3] // 2 # top left y
    y[:, 2] = x[:, 0] + x[:, 2] // 2 # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] // 2  # bottom right y

    y = y.astype(np.int)

    y[:, 0] = np.max(np.vstack((y[:, 0], zero_array)),axis=0)  # top left x
    y[:, 1] = np.max(np.vstack((y[:, 1], zero_array)),axis=0)  # top left y
    y[:, 2] = np.min(np.vstack((y[:, 2], width_array)),axis=0)  # bottom right x
    y[:, 3] = np.min(np.vstack((y[:, 3], height_array)),axis=0)  # bottom right y

    return y