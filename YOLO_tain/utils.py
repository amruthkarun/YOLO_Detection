import numpy as np
import os
import xml.etree.ElementTree as ET
import tensorflow as tf
import copy
import cv2
import matplotlib.pyplot as plt
#LABELS = ['nodule', 'no-nodule']
LABELS = ['nodule', 'no-nodule']
CLASS  = len(LABELS)
IMAGE_H, IMAGE_W = 416, 416
GRID_H,  GRID_W  = 13 , 13
BOX              = 5
CLASS            = len(LABELS)
CLASS_WEIGHTS    = np.ones(CLASS, dtype='float32')
OBJ_THRESHOLD    = 0.3
NMS_THRESHOLD    = 0.4 
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
ANCHORS          = [.5, .5, .5, .5, .5,.5,.5,.5,.5,.5]
NO_OBJECT_SCALE  = 1.0
OBJECT_SCALE     = 5.0
COORD_SCALE      = 1.0
CLASS_SCALE      = 1.0

BATCH_SIZE       = 5
WARM_UP_BATCHES  = 0
TRUE_BOX_BUFFER  = 50

class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        
        self.c     = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
            
        return self.score

class WeightReader:
    def __init__(self, weight_file):
        self.offset = 4
        self.all_weights = np.fromfile(weight_file, dtype='float32')
        
    def read_bytes(self, size):
        self.offset = self.offset + size
        return self.all_weights[self.offset-size:self.offset]
    
    def reset(self):
        self.offset = 4

def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])  
    
    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    union = w1*h1 + w2*h2 - intersect
    
    return float(intersect) / union

def draw_boxes(image,filename, boxes, labels):
    image_h, image_w, _ = image.shape
    print(filename)
    box_info=[]
    if boxes == []:
       print('in save empty')
       np.savetxt(filename + ".txt", boxes)
    else:
       for box in boxes:
            print('in boxes')
            xmin = int(box.xmin*image_w)
            ymin = int(box.ymin*image_h)
            xmax = int(box.xmax*image_w)
            ymax = int(box.ymax*image_h)
            print(xmin)
            print(ymin)
            print(xmax)
            print(ymax)
            cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (0,255,0), 3)
            cv2.putText(image, 
                        labels[box.get_label()] + ' ' + str(box.get_score()), 
                        (xmin, ymin - 13), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1e-3 * image_h, 
                        (0,255,0), 2)  
            # pred_boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes]) 
            box_i=np.array([30, 0.00, 0, 0.00, xmin, ymin, xmax, ymax, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, (box.get_score()), 0.00, 0])
            print(box_i)
            box_info.append(box_i)
            print(box_info)
       np.savetxt(filename + ".txt", box_info,fmt=['%d','%f','%d','%f','%d','%d','%d','%d','%f','%f','%f','%f','%f','%f','%f','%f','%f','%d'],delimiter=' ', newline='\r\n')    
    return image          
        
def decode_netout(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []
    
    # decode the output by the network
    netout[..., 4]  = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold
    print(grid_h)
    print(grid_w)
    print(range(nb_box))
    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row,col,b,5:]
                #print(classes)
                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row,col,b,:4]

                    x = (col + _sigmoid(x)) / grid_w # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h # unit: image height
                    confidence = netout[row,col,b,4]
                    
                    box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, confidence, classes)
                    
                    boxes.append(box)
    print(nb_class)
    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            
            if boxes[index_i].classes[c] == 0: 
                continue
            else:
                for j in range(i+1, len(sorted_indices)):
                    index_j = sorted_indices[j]
                    
                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0
                        
    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]
    
    return boxes    

def compute_overlap(a, b):
    """
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    Parameters
    ----------
    a: (N, 4) ndarray of float
    b: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])

    iw = np.minimum(np.expand_dims(a[:, 2], axis=1), b[:, 2]) - np.maximum(np.expand_dims(a[:, 0], 1), b[:, 0])
    ih = np.minimum(np.expand_dims(a[:, 3], axis=1), b[:, 3]) - np.maximum(np.expand_dims(a[:, 1], 1), b[:, 1])

    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), axis=1) + area - iw * ih

    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua  
    
def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recall, [1.]))
    mpre = np.concatenate(([0.], precision, [0.]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap      
        
def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
             return 0
        else:
            return min(x2,x4) - x3          

def _sigmoid(x):
    return 1. / (1. + np.exp(-x))

def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)
    
    if np.min(x) < t:
        x = x/np.min(x)*t
        
    e_x = np.exp(x)
    
    return e_x / e_x.sum(axis, keepdims=True)
    
def evaluate(model, 
             generator, 
             iou_threshold=0.4,
             obj_thresh=0.3,
             nms_thresh=0.4,
             net_h=416,
             net_w=416,
             save_path='E:\TED-LungNoduleDetection\keras-yolo2-master\Detections'):
    """ Evaluate a given dataset using a given model.
    code originally from https://github.com/fizyr/keras-retinanet

    # Arguments
        model           : The model to evaluate.
        generator       : The generator that represents the dataset to evaluate.
        iou_threshold   : The threshold used to consider when a detection is positive or negative.
        obj_thresh      : The threshold used to distinguish between object and non-object
        nms_thresh      : The threshold used to determine whether two detections are duplicates
        net_h           : The height of the input image to the model, higher value results in better accuracy
        net_w           : The width of the input image to the model
        save_path       : The path to save images with visualized detections to.
    # Returns
        A dict mapping class names to mAP scores.
    """    
    # gather all detections and annotations
    all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
#    print(all_detections)
#    print(all_annotations)
    
#    print(range(generator.size()))
    for i in range(generator.size()):
        raw_image = [generator.load_image(i)]
        
        # make the boxes and the labels
        pred_boxes = get_yolo_boxes(model, raw_image, net_h, net_w, ANCHORS, obj_thresh, nms_thresh)[0]
#        print('pred_boxes')
#        print(pred_boxes)
        score = np.array([box.get_score() for box in pred_boxes])
#        print('score')
        print(score)
        pred_labels = np.array([box.label for box in pred_boxes])
#        print('pred_labels')
#        print(pred_labels)
        if pred_labels == []:
            pred_labels = 1;
#        print('length of pred boxes')
#        print(len(pred_boxes))
        if len(pred_boxes) > 0:
            pred_boxes = np.array([[box.xmin, box.ymin, box.xmax, box.ymax, box.get_score()] for box in pred_boxes]) 
        else:
            pred_boxes = np.array([[0.0,0.0,0.0,0.0,0.0]])  
                       
#        print(pred_boxes)
        # sort the boxes and the labels according to scores
        score_sort = np.argsort(-score)
        pred_labels = pred_labels[score_sort]
        pred_boxes  = pred_boxes[score_sort]
        
        # copy detections to all_detections
#        print('range_generator_num_classes')
#        print(range(generator.num_classes()))
        for label in range(generator.num_classes()): #(0,1):
#            print('in label in num classes')
#            print(pred_labels)
            all_detections[i][label] = pred_boxes[pred_labels == label, :]
            print(all_detections[i][label])
            np.savetxt("detect" + str(i) +str(label) + ".txt", all_detections[i][label], fmt=['%f','%f','%f','%f','%f'],delimiter='  ', newline='\r\n')
            
        annotations = generator.load_annotation(i)
        
        # copy annotations to all_annotations
        for label in range(generator.num_classes()):
            all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
            np.savetxt("groundtruth" + str(i) +str(label) + ".txt", all_annotations[i][label], fmt=['%d','%d','%d','%d'],delimiter='  ', newline='\r\n')
#            print(pred_boxes)
#       print(all_detections)
#        print(all_annotations[i])
#            print('---over---')
#        image = cv2.imread('F:/LIDC/PositiveImages/val/images/0296_1_0139_im_000074.png')
#        cv2.imshow('Image Frame', image)
#        cv2.waitKey(0)
#        cv2.destroyAllWindows()    
#    for i in range(generator.size()):    
#        print(all_detections[i])
#        print(all_annotations[i])
    
    # compute mAP by comparing all detections and all annotations
    average_precisions = {}
    
    for label in range(generator.num_classes()):
        print('for each label.....')
        false_positives = np.zeros((0,))
        no_nodules_true = np.zeros((0,))
        true_positives  = np.zeros((0,))
        scores          = np.zeros((0,))
        num_annotations = 0.0
        no_nodules=0.0
        for i in range(generator.size()):
            detections           = all_detections[i][label]
#            print(detections[1:])
            annotations          = all_annotations[i][label]
#            print(annotations[1:])
            num_annotations     += annotations.shape[0]
            #print(num_annotations)
            detected_annotations = []
            if num_annotations == 1.0:
                if detections.shape[0] == 0:
                   no_nodules += 1;
             #      print('in no nodules true')
                   no_nodules_true  = np.append(no_nodules_true, 1)
            for d in detections:
                scores = np.append(scores, d[4])
                print(scores)
                if annotations.shape[0] == 0:
                    false_positives = np.append(false_positives, 1)
                    #false_negatives = np.append(false_negatives, 1)
                    true_positives  = np.append(true_positives, 0)
                    continue

                overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                assigned_annotation = np.argmax(overlaps, axis=1)
                max_overlap         = overlaps[0, assigned_annotation]

                if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                    false_positives = np.append(false_positives, 0)
                    true_positives  = np.append(true_positives, 1)
                    detected_annotations.append(assigned_annotation)
                else:
                    false_positives = np.append(false_positives, 1)
                    true_positives  = np.append(true_positives, 0)
                print(false_positives)
                print(true_positives)
        # no annotations -> AP for this class is 0 (is this correct?)
        if num_annotations == 0:
            average_precisions[label] = 0
            continue
#        print(false_positives)
#        print(true_positives)
#        print(no_nodules_true)
        # sort by score
        #print('---sort by score--')
        indices         = np.argsort(-scores)
        false_positives = false_positives[indices]
        true_positives  = true_positives[indices]
        
        # compute false positives and true positives
        print('---false positives and true positives--')
        false_positives = np.cumsum(false_positives)
        true_positives  = np.cumsum(true_positives)
#        print(false_positives)
#        print(true_positives)
        # compute recall and precision
        print('---compute recall and precision--')
        recall    = true_positives / num_annotations
        precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
#        print(precision)
#        print(recall)
        # compute average precision
        average_precision  = compute_ap(recall, precision)  
        average_precisions[label] = average_precision
    
    print(false_positives)
    print(true_positives)
    print(no_nodules_true)
    print(average_precision)
    print(average_precisions)
    return average_precisions    

def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    if (float(net_w)/image_w) < (float(net_h)/image_h):
        new_w = net_w
        new_h = (image_h*net_w)/image_w
    else:
        new_h = net_w
        new_w = (image_w*net_h)/image_h
        
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)
        
def do_nms(boxes, nms_thresh):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
        
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0: continue

            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]

                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0

def preprocess_input(image, net_h, net_w):
    new_h, new_w, _ = image.shape

    # determine the new size of the image
    if (float(net_w)/new_w) < (float(net_h)/new_h):
        new_h = (new_h * net_w)//new_w
        new_w = net_w
    else:
        new_w = (new_w * net_h)//new_h
        new_h = net_h

    # resize the image to the new size
    resized = cv2.resize(image[:,:,::-1]/1., (new_w, new_h))

    # embed the image into the standard letter box
    new_image = np.ones((net_h, net_w, 3)) * 0.5
    new_image[(net_h-new_h)//2:(net_h+new_h)//2, (net_w-new_w)//2:(net_w+new_w)//2, :] = resized
    new_image = np.expand_dims(new_image, 0)

    return new_image

def normalize(image):
   # return image/255.
    return image/1. 

def get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
    image_h, image_w, _ = images[0].shape
    nb_images           = len(images)
    batch_input         = np.zeros((nb_images, net_h, net_w, 3))
    print(batch_input.shape)
    # preprocess the input
    for i in range(nb_images):
        batch_input[i] = preprocess_input(images[i], net_h, net_w)        
    print(batch_input.shape)
    print('sss')
    print(nb_images)
    # run the prediction
    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
    image=batch_input;
    input_image=image
#    input_image = input_image[:,:,::-1]
#    input_image = np.expand_dims(input_image, 0)
    print(input_image.shape)
    
    batch_output  = model.predict([input_image, dummy_array])
    print(batch_output[0].shape)
    
    boxes = decode_netout(batch_output[0], 
                          obj_threshold=OBJ_THRESHOLD,
                          nms_threshold=NMS_THRESHOLD,
                          anchors=ANCHORS, 
                          nb_class=CLASS)
    # correct the sizes of the bounding boxes
    correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
    #print(boxes) 
    #plt.figure(figsize=(10,10))          
    #image = draw_boxes(images[0], boxes, labels=LABELS)
    #plt.imshow(image[:,:,::-1]); plt.show()
#   batch_output = model.predict_on_batch(batch_input)
    batch_boxes  = [None]*nb_images
    batch_boxes[0] = boxes

    return batch_boxes           
    
    
#def new_get_yolo_boxes(model, images, net_h, net_w, anchors, obj_thresh, nms_thresh):
#    image_h, image_w, _ = images[0].shape
#    nb_images           = len(images)
#    batch_input         = np.zeros((nb_images, net_h, net_w, 3))
#    ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
#    # preprocess the input
#    for i in range(nb_images):
#        batch_input[i] = preprocess_input(images[i], net_h, net_w)  
#        ###batch_input[i] = cv2.resize(images[i], (416, 416))
#        
#    print('sss')
#    print(len(batch_input))
#    print(nb_images)
#    print(batch_input[0].shape)    
#    print(i)
#    cv2.waitKey(0)
#    # run the prediction
#    #batch_output = model.predict_on_batch(batch_input)
#    batch_boxes  = [None]*nb_images
#    dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
#
#    image=batch_input[0];
#    #input_image = cv2.resize(image, (416, 416))
#    input_image=image
#    #input_image = input_image / 255.
#    input_image = input_image[:,:,::-1]
#    input_image = np.expand_dims(input_image, 0)
#    print(input_image.shape)
#    batch_output  = model.predict([input_image, dummy_array])
#    cv2.imshow('Image Frame inside utils', image)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows()
##    boxes = decode_netout(batch_output[0], 
##                          obj_threshold=OBJ_THRESHOLD,
##                          nms_threshold=NMS_THRESHOLD,
##                          anchors=ANCHORS, 
##                          nb_class=CLASS)
##    print('After decode..........')
##    print(boxes)
#    #image = cv2.imread('F:/LIDC/PositiveImages/val/images/0296_1_0139_im_000074.png')
#    #dummy_array = np.zeros((1,1,1,1,TRUE_BOX_BUFFER,4))
#    #plt.figure(figsize=(10,10))
#    #input_image = cv2.resize(image, (416, 416))
#    #print(input_image.shape)
#    #input_image = input_image / 255.
#    #input_image = input_image[:,:,::-1]
#    #input_image = np.expand_dims(input_image, 0)
#    #print(input_image.shape)
#    #cv2.waitKey(0)
#    #netout = model.predict([input_image, dummy_array])
#    #print(netout[0].shape)
#    #batch_output  = model.predict_on_batch([input_image, dummy_array])
#    #print(batch_output[0])
#    #print(batch_output.shape)
#    #print(range(nb_images))
#    #cv2.imshow('Image Frame', image)
#    #cv2.waitKey(0)
#    #cv2.destroyAllWindows()
#    for i in range(nb_images):
#         print('.....yolo......')
##        #print(batch_output[0][i])
#         yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
#         yolos = batch_output[0]
#         print(batch_output[0].shape)
#         print(len(yolos))
#         boxes = []
##        cv2.waitKey(0)
#         #netout = model.predict([input_image, dummy_array])
##         boxes = decode_netout(batch_output[0], 
##                               obj_threshold=OBJ_THRESHOLD,
##                               nms_threshold=NMS_THRESHOLD,
##                               anchors=ANCHORS, 
##                               nb_class=CLASS)
#         for i in range(nb_images):
#           yolos = [batch_output[0][i], batch_output[1][i], batch_output[2][i]]
#           boxes = []
#
#        # decode the output of the network
#        for j in range(len(yolos)):
#            yolo_anchors = anchors[(2-j)*6:(3-j)*6] # config['model']['anchors']
#            boxes += decode_netout(yolos[j], yolo_anchors, obj_thresh, net_h, net_w)
#
#        # correct the sizes of the bounding boxes
#        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
#
#        # suppress non-maximal boxes
#        do_nms(boxes, nms_thresh)        
#           
#        batch_boxes[i] = boxes
#
#    return batch_boxes     
##        
#         print(boxes)  
#         input_image = draw_boxes(image, boxes, labels=LABELS)
#         plt.imshow(image[:,:,::-1]); plt.show()
##        # correct the sizes of the bounding boxes
##        correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w)
##        print(boxes)
##        # suppress non-maximal boxes
##        do_nms(boxes, nms_thresh)        
##        print(boxes)   
#         batch_boxes[i] = boxes
#
#    return batch_boxes   
    
def eval_decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2]
    nb_box = 3
    netout = netout.reshape((grid_h, grid_w, nb_box, -1))
    nb_class = netout.shape[-1] - 5

    boxes = []

    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4]   = _sigmoid(netout[..., 4])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_thresh

    for i in range(grid_h*grid_w):
        row = i // grid_w
        col = i % grid_w
        
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[row, col, b, 4]
            
            if(objectness <= obj_thresh): continue
            
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[row,col,b,:4]

            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height  
            
            # last elements are class probabilities
            classes = netout[row,col,b,5:]
            
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)

            boxes.append(box)

    return boxes
