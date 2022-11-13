import time
import cv2
import numpy as np
import onnxruntime
from imread_from_url import imread_from_url

from .utils_detector import non_max_suppression, xywh2xyxy

anchors = [10,14,  23,27,  37,58,  81,82,  135,169,  344,319]
class YoloV5s():

    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres

        # Initialize model
        self.model = self.initialize_model(model_path)

    def __call__(self, image):

        return self.detect_objects(image)

    def initialize_model(self, model_path):

        self.session = onnxruntime.InferenceSession(model_path,
                                                    providers=['CUDAExecutionProvider',
                                                               'CPUExecutionProvider'])

        # Get model info
        self.getModel_input_details()
        self.getModel_output_details()

    def detect_objects(self, image):

        input_tensor = self.prepare_input(image)

        output = self.inference(input_tensor)

        boxes, scores = self.process_output(output)

        return boxes, scores

    def prepare_input(self, img):

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.img_height, self.img_width, self.img_channels = img.shape

        img_input = cv2.resize(img, (self.input_width, self.input_height))
                
        img_input = img_input/ 255.0
        # img_input = img_input.transpose(2, 0, 1)
        img_input = np.expand_dims(img_input, 0)     

        return img_input.astype(np.float32)

    def inference(self, input_tensor):

        return self.session.run(self.output_names, {self.input_name: input_tensor})[0]

    def process_output(self, output):


        output = np.squeeze(output)

        # Filter boxes with low confidence
        output = output[output[:,4] > self.conf_thres]

        # Filter person class only
        classId = np.argmax(output[:,5:], axis=1)
        output = output[classId == 0]

        boxes = output[:,:4]
        boxes[:, 0] *= self.img_width
        boxes[:, 1] *= self.img_height 
        boxes[:, 2] *= self.img_width  
        boxes[:, 3] *= self.img_height 

        # Keep boxes only with positive width and height
        boxes = boxes[np.logical_or(boxes[:,2] > 0, boxes[:,3] > 0)]

        scores = output[:,4]
        boxes = xywh2xyxy(boxes, self.img_width, self.img_height).astype(int)

        box_ids = non_max_suppression(boxes, scores, self.iou_thres)

        if box_ids.shape[0] == 0:
            return None, None

        scores = scores[box_ids]
        boxes = boxes[box_ids,:]

        return boxes, scores

    def getModel_input_details(self):

        self.input_name = self.session.get_inputs()[0].name

        self.input_shape = self.session.get_inputs()[0].shape
        self.input_height = self.input_shape[1]
        self.input_width = self.input_shape[2]
        self.channels = self.input_shape[3]

    def getModel_output_details(self):

        model_outputs = self.session.get_outputs()
        self.output_names = []
        self.output_names.append(model_outputs[0].name)

    @staticmethod
    def draw_detections(img, boxes, scores):

        if boxes is None:
            return img

        for box, score in zip(boxes, scores):
            img = cv2.rectangle(img, (int(box[0]),int(box[1])), (int(box[2]),int(box[3])), (255,191,0), 1)

            cv2.putText(img, str(int(100*score)) + '%', (int(box[0]),int(box[1])), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,191,0), 1, cv2.LINE_AA)

        return img

if __name__ == '__main__':

    model_path='../models/model_float32.onnx'  
    image = imread_from_url("https://upload.wikimedia.org/wikipedia/commons/thumb/9/91/Bruce_McCandless_II_during_EVA_in_1984.jpg/768px-Bruce_McCandless_II_during_EVA_in_1984.jpg")

    object_detector = YoloV5s(model_path)

    boxes, scores = object_detector(image)  

    image = YoloV5s.draw_detections(image, boxes, scores)
   
    cv2.namedWindow("Detected people", cv2.WINDOW_NORMAL)
    cv2.imshow("Detected people", image)
    cv2.waitKey(0)