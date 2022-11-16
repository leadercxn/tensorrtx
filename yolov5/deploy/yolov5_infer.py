import ctypes
import os
import shutil
import random
import sys
import threading
import time
import cv2
import numpy as np
import torch
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import argparse
from multiprocessing import Process
import subprocess
import threading
from dataloaders import LoadImages,LoadWebcam,LoadStreams


########################################################
## ObjectDection
########################################################

CONF_THRESH = 0.5
IOU_THRESHOLD = 0.4

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return

    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

class YoLov5TRT_ObjectDection(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []

        for binding in engine:
            print('bingding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings
        # Do image preprocess
        batch_image_raw = []
        batch_origin_h = []
        batch_origin_w = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])
        '''
        for i, image_raw in enumerate(raw_image_generator):
            input_image, image_raw, origin_h, origin_w = self.preprocess_image(image_raw)
            batch_image_raw.append(image_raw)
            batch_origin_h.append(origin_h)
            batch_origin_w.append(origin_w)
            np.copyto(batch_input_image[i], input_image)
        '''
        input_image, image_raw, origin_h, origin_w = self.preprocess_image(raw_image_generator)
        batch_image_raw.append(image_raw)
        batch_origin_h.append(origin_h)
        batch_origin_w.append(origin_w)
        np.copyto(batch_input_image, input_image)

        batch_input_image = np.ascontiguousarray(batch_input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        result = context.execute_async(batch_size=self.batch_size, bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        for i in range(self.batch_size):
            result_boxes, result_scores, result_classid = self.post_process(
                output[i * 6001: (i + 1) * 6001], batch_origin_h[i], batch_origin_w[i])
        return result_boxes, result_scores, result_classid, result   

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        
    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)
        
    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_image(self, raw_bgr_image):
        """
        description: Convert BGR image to RGB,
                     resize and pad it to target size, normalize to [0,1],
                     transform to NCHW format.
        param:
            input_image_path: str, image path
        return:
            image:  the processed image
            image_raw: the original image
            h: original height
            w: original width
        """
        image_raw = raw_bgr_image
        h, w, c = image_raw.shape
        image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2RGB)
        # Calculate widht and height and paddings
        r_w = self.input_w / w
        r_h = self.input_h / h
        if r_h > r_w:
            tw = self.input_w
            th = int(r_w * h)
            tx1 = tx2 = 0
            ty1 = int((self.input_h - th) / 2)
            ty2 = self.input_h - th - ty1
        else:
            tw = int(r_h * w)
            th = self.input_h
            tx1 = int((self.input_w - tw) / 2)
            tx2 = self.input_w - tw - tx1
            ty1 = ty2 = 0
        # Resize the image with long side while maintaining ratio
        image = cv2.resize(image, (tw, th))
        # Pad the short side with (128,128,128)
        image = cv2.copyMakeBorder(
            image, ty1, ty2, tx1, tx2, cv2.BORDER_CONSTANT, None, (128, 128, 128)
        )
        image = image.astype(np.float32)
        # Normalize to [0,1]
        image /= 255.0
        # HWC to CHW format:
        image = np.transpose(image, [2, 0, 1])
        # CHW to NCHW format
        image = np.expand_dims(image, axis=0)
        # Convert the image to row-major order, also known as "C order":
        image = np.ascontiguousarray(image)
        return image, image_raw, h, w

    def xywh2xyxy(self, origin_h, origin_w, x):
        """
        description:    Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        param:
            origin_h:   height of original image
            origin_w:   width of original image
            x:          A boxes numpy, each row is a box [center_x, center_y, w, h]
        return:
            y:          A boxes numpy, each row is a box [x1, y1, x2, y2]
        """
        y = np.zeros_like(x)
        r_w = self.input_w / origin_w
        r_h = self.input_h / origin_h
        if r_h > r_w:
            y[:, 0] = x[:, 0] - x[:, 2] / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2 - (self.input_h - r_w * origin_h) / 2
            y /= r_w
        else:
            y[:, 0] = x[:, 0] - x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 2] = x[:, 0] + x[:, 2] / 2 - (self.input_w - r_h * origin_w) / 2
            y[:, 1] = x[:, 1] - x[:, 3] / 2
            y[:, 3] = x[:, 1] + x[:, 3] / 2
            y /= r_h

        return y

    def post_process(self, output, origin_h, origin_w):
        """
        description: postprocess the prediction
        param:
            output:     A numpy likes [num_boxes,cx,cy,w,h,conf,cls_id, cx,cy,w,h,conf,cls_id, ...] 
            origin_h:   height of original image
            origin_w:   width of original image
        return:
            result_boxes: finally boxes, a boxes numpy, each row is a box [x1, y1, x2, y2]
            result_scores: finally scores, a numpy, each element is the score correspoing to box
            result_classid: finally classid, a numpy, each element is the classid correspoing to box
        """
        # Get the num of boxes detected
        num = int(output[0])
        # Reshape to a two dimentional ndarray
        pred = np.reshape(output[1:], (-1, 6))[:num, :]
        # Do nms
        boxes = self.non_max_suppression(pred, origin_h, origin_w, conf_thres=CONF_THRESH, nms_thres=IOU_THRESHOLD)
        result_boxes = boxes[:, :4] if len(boxes) else np.array([])
        result_scores = boxes[:, 4] if len(boxes) else np.array([])
        result_classid = boxes[:, 5] if len(boxes) else np.array([])
        return result_boxes, result_scores, result_classid

    def bbox_iou(self, box1, box2, x1y1x2y2=True):
        """
        description: compute the IoU of two bounding boxes
        param:
            box1: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))
            box2: A box coordinate (can be (x1, y1, x2, y2) or (x, y, w, h))            
            x1y1x2y2: select the coordinate format
        return:
            iou: computed iou
        """
        if not x1y1x2y2:
            # Transform from center and width to exact coordinates
            b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
            b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
            b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
            b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
        else:
            # Get the coordinates of bounding boxes
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

        # Get the coordinates of the intersection rectangle
        inter_rect_x1 = np.maximum(b1_x1, b2_x1)
        inter_rect_y1 = np.maximum(b1_y1, b2_y1)
        inter_rect_x2 = np.minimum(b1_x2, b2_x2)
        inter_rect_y2 = np.minimum(b1_y2, b2_y2)
        # Intersection area
        inter_area = np.clip(inter_rect_x2 - inter_rect_x1 + 1, 0, None) * \
                     np.clip(inter_rect_y2 - inter_rect_y1 + 1, 0, None)
        # Union Area
        b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
        b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

        iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

        return iou

    def non_max_suppression(self, prediction, origin_h, origin_w, conf_thres=0.5, nms_thres=0.4):
        """
        description: Removes detections with lower object confidence score than 'conf_thres' and performs
        Non-Maximum Suppression to further filter detections.
        param:
            prediction: detections, (x1, y1, x2, y2, conf, cls_id)
            origin_h: original image height
            origin_w: original image width
            conf_thres: a confidence threshold to filter detections
            nms_thres: a iou threshold to filter detections
        return:
            boxes: output after nms with the shape (x1, y1, x2, y2, conf, cls_id)
        """
        # Get the boxes that score > CONF_THRESH
        boxes = prediction[prediction[:, 4] >= conf_thres]
        # Trandform bbox from [center_x, center_y, w, h] to [x1, y1, x2, y2]
        boxes[:, :4] = self.xywh2xyxy(origin_h, origin_w, boxes[:, :4])
        # clip the coordinates
        boxes[:, 0] = np.clip(boxes[:, 0], 0, origin_w -1)
        boxes[:, 2] = np.clip(boxes[:, 2], 0, origin_w -1)
        boxes[:, 1] = np.clip(boxes[:, 1], 0, origin_h -1)
        boxes[:, 3] = np.clip(boxes[:, 3], 0, origin_h -1)
        # Object confidence
        confs = boxes[:, 4]
        # Sort by the confs
        boxes = boxes[np.argsort(-confs)]
        # Perform non-maximum suppression
        keep_boxes = []
        while boxes.shape[0]:
            large_overlap = self.bbox_iou(np.expand_dims(boxes[0, :4], 0), boxes[:, :4]) > nms_thres
            label_match = boxes[0, -1] == boxes[:, -1]
            # Indices of boxes with lower confidence scores, large IOUs and matching labels
            invalid = large_overlap & label_match
            keep_boxes += [boxes[0]]
            boxes = boxes[~invalid]
        boxes = np.stack(keep_boxes, 0) if len(keep_boxes) else np.array([])
        return boxes

class inferThread_ObjectDection(threading.Thread):
    def __init__(self, yolov5_wrapper, raw_image):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.raw_image = raw_image

        # load coco labels
        self.categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]
        

    def run(self):
        result_boxes, result_scores, result_classid, ret = self.yolov5_wrapper.infer(self.raw_image)

        if ret and ret != 'false':
            # print(str(obj))
            # print('real infer, time->{:.2f}ms'.format(t * 1000))
            for j in range(len(result_boxes)):
                box = result_boxes[j]
                plot_one_box(
                    box,
                    self.raw_image,
                    label="{}:{:.2f}".format(
                        self.categories[int(result_classid[j])], result_scores[j]
                    ),
                )

            cv2.imshow('yolov5-detection', self.raw_image)
            cv2.waitKey(20)
        else:
            obj = {"boxes":[],"scores":[],"labels":[]}
            print("No suit categories")




########################################################
## Classification
########################################################

with open("../imagenet_classes.txt") as f:
    classes = [line.strip() for line in f.readlines()]

class YoLov5TRT_Classification(object):
    """
    description: A YOLOv5 class that warps TensorRT ops, preprocess and postprocess ops.
    """

    def __init__(self, engine_file_path):
        # Create a Context on this device,
        self.ctx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.INFO)
        runtime = trt.Runtime(TRT_LOGGER)

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        host_inputs = []
        cuda_inputs = []
        host_outputs = []
        cuda_outputs = []
        bindings = []
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        for binding in engine:
            print('binding:', binding, engine.get_binding_shape(binding))
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                self.input_w = engine.get_binding_shape(binding)[-1]
                self.input_h = engine.get_binding_shape(binding)[-2]
                host_inputs.append(host_mem)
                cuda_inputs.append(cuda_mem)
            else:
                host_outputs.append(host_mem)
                cuda_outputs.append(cuda_mem)

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.host_inputs = host_inputs
        self.cuda_inputs = cuda_inputs
        self.host_outputs = host_outputs
        self.cuda_outputs = cuda_outputs
        self.bindings = bindings
        self.batch_size = engine.max_batch_size

    def infer(self, raw_image_generator):
        threading.Thread.__init__(self)
        # Make self the active context, pushing it on top of the context stack.
        self.ctx.push()
        # Restore
        stream = self.stream
        context = self.context
        engine = self.engine
        host_inputs = self.host_inputs
        cuda_inputs = self.cuda_inputs
        host_outputs = self.host_outputs
        cuda_outputs = self.cuda_outputs
        bindings = self.bindings

        # Do image preprocess
        batch_image_raw = []
        batch_input_image = np.empty(shape=[self.batch_size, 3, self.input_h, self.input_w])

        '''
        for i, image_raw in enumerate(raw_image_generator):
            batch_image_raw.append(image_raw)
            input_image = self.preprocess_cls_image(image_raw)
            np.copyto(batch_input_image[i], input_image)
        '''

        input_image = self.preprocess_cls_image(raw_image_generator)
        np.copyto(batch_input_image, input_image)

        batch_input_image = np.ascontiguousarray(batch_input_image)
        raw_image = np.ascontiguousarray(input_image)

        # Copy input image to host buffer
        np.copyto(host_inputs[0], batch_input_image.ravel())
        start = time.time()
        # Transfer input data  to the GPU.
        cuda.memcpy_htod_async(cuda_inputs[0], host_inputs[0], stream)
        # Run inference.
        context.execute_async(batch_size=self.batch_size,
                              bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        cuda.memcpy_dtoh_async(host_outputs[0], cuda_outputs[0], stream)
        # Synchronize the stream
        stream.synchronize()
        end = time.time()
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()
        # Here we use the first row of output in that batch_size = 1
        output = host_outputs[0]
        # Do postprocess
        for i in range(self.batch_size):
            classes_ls, predicted_conf_ls, category_id_ls = self.postprocess_cls(output)

        return str(classes_ls)

        '''
            cv2.putText(raw_image, str(classes_ls), (10, 40), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
            print(classes_ls, predicted_conf_ls)
            cv2.imshow('yolov5-cls', raw_image)  # 显示读取到的这一帧画面
            cv2.waitKey(25)

        return batch_image_raw, end - start
        '''

    def destroy(self):
        # Remove any context from the top of the context stack, deactivating it.
        self.ctx.pop()

    def get_raw_image(self, image_path_batch):
        """
        description: Read an image from image path
        """
        for img_path in image_path_batch:
            yield cv2.imread(img_path)

    def get_raw_image_zeros(self, image_path_batch=None):
        """
        description: Ready data for warmup
        """
        for _ in range(self.batch_size):
            yield np.zeros([self.input_h, self.input_w, 3], dtype=np.uint8)

    def preprocess_cls_image(self, input_img):
        im = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        im = cv2.resize(im, (self.input_h, self.input_w))
        im = np.float32(im)
        im /= 255.0
        im -= self.mean
        im /= self.std
        im = im.transpose(2, 0, 1)
        # prepare batch
        batch_data = np.expand_dims(im, axis=0)

        return batch_data

    def postprocess_cls(self, output_data):
        classes_ls = []
        predicted_conf_ls = []
        category_id_ls = []
        output_data = output_data.reshape(self.batch_size, -1)
        output_data = torch.Tensor(output_data)
        p = torch.nn.functional.softmax(output_data, dim=1)
        score, index = torch.topk(p, 3)
        for ind in range(index.shape[0]):
            input_category_id = index[ind][0].item()  # 716
            category_id_ls.append(input_category_id)
            predicted_confidence = score[ind][0].item()
            predicted_conf_ls.append(predicted_confidence)
            classes_ls.append(classes[input_category_id])
        return classes_ls, predicted_conf_ls, category_id_ls

class inferThread_Classification(threading.Thread):
    def __init__(self, yolov5_wrapper, raw_image):
        threading.Thread.__init__(self)
        self.yolov5_wrapper = yolov5_wrapper
        self.raw_image = raw_image

    def run(self):
        cls_type = self.yolov5_wrapper.infer(self.raw_image)

        # 标注释
        cv2.putText(self.raw_image, cls_type, (10, 60), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
        # 显示读取到的这一帧画面
        cv2.imshow('yolov5-cls', self.raw_image)
        cv2.waitKey(30)



########################################################
## Main
########################################################
def parse_args():
    parser = argparse.ArgumentParser(description='show any help')
    parser.add_argument('-e', '--engine'    , required=True, help='Input engine (.engine) file path (required)')
    parser.add_argument('-i', '--interface' , required=True, help='{usb,csi,rtsp}, choose video-stream source')
    parser.add_argument('-d', '--device'    , required=True, help='interface = usb, device = 0, mean /dev/video0; interface = csi, device = 0, mean sensor-id = 0; interface = rtsp, device = 192.168.111.118, mean rtsp-server = 192.168.111.118;')
    parser.add_argument('-p', '--port'      , help='when interface = rtsp, port = rtsp-server port')
    parser.add_argument('-t', '--type'      , default='detect' ,help='{detect,cls}, determines the model is detection/classification, default detect')
    parser.add_argument('-l', '--lib'       , help='when type = detect ,choose libmyplugins.so path')

    args = parser.parse_args()
    
    return args

if __name__ == "__main__":
    args = parse_args()

    engine_file_path = args.engine
    interface = args.interface
    device = args.device
    model_type = args.type

    if model_type == 'detect':
        print("model_type={}".format(model_type))
        if args.lib :
            print("libmyplugins={}".format(args.lib))
            PLUGIN_LIBRARY = str(args.lib)
            ctypes.CDLL(PLUGIN_LIBRARY)
        else :
            sys.exit("ERROR: Please choose libmyplugins.so in detection model")
    elif model_type == 'cls':
        print("model_type={}".format(model_type))
    else :
        sys.exit("ERROR: Invalid parameter --type")


    if interface == 'rtsp' :
        print("interface={}".format(interface))
        if not args.port:
            sys.exit("ERROR: Please choose valid rtsp port")
        source = ( "rtspsrc location=rtsp://%s:%s/live ! rtph264depay ! "
                   "h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw,width=800,height=480,format=BGRx ! "
                   "videoconvert ! video/x-raw,format=BGR ! appsink "
                    % (device, args.port) )
    elif interface == 'usb':
        print("interface={}".format(interface))

        source = ("v4l2src device=/dev/video%s ! image/jpeg,framerate=30/1,width=640, height=480,type=video ! "
                  "jpegdec ! videoconvert ! video/x-raw ! appsink"
                    % (device) )
    elif interface == 'csi':
        print("interface={}".format(interface))

        source = ("nvarguscamerasrc sensor-id=%s ! video/x-raw(memory:NVMM), width=(int)640, height=(int)480, framerate=(fraction)30/1 ! "
                  "nvvidconv flip-method=0 ! video/x-raw, width=(int)640, height=(int)480, format=(string)BGRx ! "
                  "videoconvert ! video/x-raw, format=(string)BGR ! appsink"
                    % (device) )
    else :
        sys.exit("ERROR: Invalid parameter --interface")

    dataloader = LoadStreams(source)

    if model_type == 'detect':
        yolov5_wrapper = YoLov5TRT_ObjectDection(engine_file_path)

    else :
        yolov5_wrapper = YoLov5TRT_Classification(engine_file_path)


    try:
        for path, im, im0s, vid_cap, s in dataloader:
            for img in im0s:
                if model_type == 'detect':
                    thread1 = inferThread_ObjectDection(yolov5_wrapper, img)
                else :
                    thread1 = inferThread_Classification(yolov5_wrapper, img)
                thread1.start()
                thread1.join()

    finally:
        # destroy the instance
        yolov5_wrapper.destroy()

####
# Usage :

## Detection
# USB Camera
# python yolov5_infer.py -e /home/seeed/github/tensorrtx/yolov5/build/yolov5m.engine -l /home/seeed/github/tensorrtx/yolov5/build/libmyplugins.so -i usb -d 1 -t detect

# CSI Camera
# python yolov5_infer.py -e /home/seeed/github/tensorrtx/yolov5/build/yolov5m.engine -l /home/seeed/github/tensorrtx/yolov5/build/libmyplugins.so -i csi -d 0 -t detect

# RTSP Camera
# python yolov5_infer.py -e /home/seeed/github/tensorrtx/yolov5/build/yolov5m.engine -l /home/seeed/github/tensorrtx/yolov5/build/libmyplugins.so -i rtsp -d 192.168.111.118 -p 8081 -t detect


## Classification
# USB Camera
# python yolov5_infer.py -e /home/seeed/github/tensorrtx/yolov5/build/yolov5m-cls.engine -i usb -d 1 -t cls

# CSI Camera
# python yolov5_infer.py -e /home/seeed/github/tensorrtx/yolov5/build/yolov5m-cls.engine -i csi -d 0 -t cls

# USB Camera
# python yolov5_infer.py -e /home/seeed/github/tensorrtx/yolov5/build/yolov5m-cls.engine -i rtsp -d 192.168.111.118 -p 8554 -t cls



###