#!/usr/bin/env python

"""
ROS Node for object detection using py-faster-rcnn
https://github.com/rbgirshick/py-faster-rcnn.git
or py-r-fcn
https://github.com/Orpine/py-R-FCN.git
"""
import roslib; roslib.load_manifest("deep_object_detection")
import rospy
from rospkg import RosPack
import sys
import os


from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from fast_rcnn.nms_wrapper import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe
#print caffe.__file__
import cv2
import argparse
from deep_object_detection.srv import *
from deep_object_detection.msg import Object, DetectedObjects
from cv_bridge import CvBridge, CvBridgeError

""" DeepObjectDetection class for object detection """
class DeepObjectDetection():

    """ The handle for object detection service requests """
    def handle_detect_objects_req(self,req):
        self.service_queue += 1
        if self.net == None:
            try:
                self.net = caffe.Net(self.prototxt, self.caffemodel, caffe.TEST)
            except:
                rospy.logerr("Error, cannot load deep_net to the GPU")
                self.net =None
                self.service_queue -=1
                return DetectObjectsResponse([])
        if self.unload_net_timer != None:
            self.unload_net_timer.shutdown()
            del self.unload_net_timer
            self.unload_net_timer = None
        self.unload_net_timer = rospy.Timer(rospy.Duration(60), self.unload_net_callback)
        bridge = CvBridge()
        results = []
        for index,image in enumerate(req.images):
            try:
                cv_image = bridge.imgmsg_to_cv2(image, "bgr8")
                if self.gpu_id >= 0:
                    caffe.set_mode_gpu()
                    caffe.set_device(self.gpu_id)
                else:
                    caffe.set_mode_cpu()

                # if confidence threshold is not set, default=0.8
                if req.confidence_threshold == 0:
                    try:
                        #rospy.logerr("Hello")
                        self.detect_objects(results,self.net,cv_image,index)
                    except:
                        rospy.logwarn("Error!! Could not execute detect_object function!! Returning empty service response")
                        self.service_queue -=1
                        return DetectObjectsResponse([])
                else:
                    try:
                        self.detect_objects(results,self.net,cv_image,index,req.confidence_threshold)
                    except ValueError, Argument:
                        rospy.logerr("%s",Argument)
                        rospy.logwarn("Error!! Could not execute detect_object function!! Returning empty service response")
                        self.service_queue-=1
                        return DetectObjectsResponse([])

		    #results.append(result)
            except CvBridgeError as e:
                rospy.logerr("CVBridge exception %s",e)
                self.service_queue -=1
                return DetectObjectsResponse([])
        self.service_queue -=1
        detectedobjects = DetectedObjects()
        detectedobjects.objects = results
        detectedobjects.observation_path = req.observation_path
        self.detectedobjectspub.publish(detectedobjects)
        return DetectObjectsResponse(results)

    def handle_getlabels_req(self,req):
        return GetLabelsResponse(self.CLASSES)

    def unload_net_callback(self,event):
        if self.net != None and self.service_queue <= 0:
            del self.net
            self.net = None
            self.service_queue =0
            self.unload_net_timer.shutdown()
            del self.unload_net_timer
            self.unload_net_timer = None
        elif self.net == None and self.service_queue != 0:
            self.service_queue = 0
            self.unload_net_timer.shutdown()
            del self.unload_net_timer
            self.unload_net_timer = None




    def detect_objects(self,results,net, image,image_index=0,conf_thresh=0.8):
        """Detect object classes in an image using pre-computed object proposals."""
        #rospy.logerr("Starting detection...")
        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        scores, boxes = im_detect(net, image)
        timer.toc()
        astr = ('Detection took {:.3f}s for '
        '{:d} object proposals').format(timer.total_time, boxes.shape[0])
        rospy.loginfo(astr)

        # Filter out the results based on confidence_threshold and then prepare the objects
        CONF_THRESH = conf_thresh
        NMS_THRESH = 0.3
        objects = []
        for cls_ind, cls in enumerate(self.CLASSES[1:]):
            cls_ind += 1 # because we skipped background
            if self.network_name.find("Res") < 0:
                cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
            else:
                cls_boxes = boxes[:, 4:8]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
            if len(inds) > 0:
                for i in inds:
                    obj = Object()
                    bbox = dets[i, :4]
                    score = dets[i, -1]
                    obj.label = cls
                    obj.x = bbox[0]
                    obj.y = bbox[1]
                    obj.width = bbox[2]-bbox[0]
                    obj.height = bbox[3]-bbox[1]
                    obj.confidence = score
                    obj.imageID = image_index
		    results.append(obj)
            #vis_detections(im, cls, dets, thresh=CONF_THRESH)
        #return objects


    def __init__(self, gpu_id=None, network_name='vgg16'):
        rp = RosPack()

        path = rp.get_path('deep_models')

        self.network_name = network_name

        self.CLASSES = ('__background__',
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa','train', 'tvmonitor')

        self.NETS = {'vgg16': ('VGG16',
                      'VGG16_faster_rcnn_final.caffemodel'),
		     'vgg16KTH': ('VGG16KTH',
                      'vgg16_faster_rcnn_iter_40000_kth2.caffemodel'),
            'zf': ('ZF',
                      'ZF_faster_rcnn_final.caffemodel'),
            'ResNet-101': ('ResNet-101',
                  'resnet101_rfcn_final.caffemodel'),
        'ResNet-50': ('ResNet-50',
                  'resnet50_rfcn_final.caffemodel'),
        'ResNet-101coco':('ResNet-101','resnet101_rfcn_coco.caffemodel')
        }
        if network_name == 'vgg16KTH':
            self.CLASSES = ('__background__',
              'chair','laptop','monitor')
        if network_name.find('coco') >= 0:
            self.CLASSES = ('__background__','person','bicycle','car','motorbike','aeroplane','bus','train','truck','boat','traffic_light','fire_hydrant','stop_sign','parking_meter','bench','bird','cat',
                            'dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports_ball','kite', 'baseball_bat',
                            'baseball_glove','skateboard','surfboard','tennis_racket','bottle','wine_glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli',
                            'carrot','hot_dog','pizza','donut','cake','chair','sofa','pottedplant','bed','diningtable','toilet','tvmonitor','laptop','mouse','remote','keyboard','cell_phone',
                            'microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy_bear','hair_drier','toothbrush')


        cfg.TEST.HAS_RPN = True  # Use RPN for proposals
        self.prototxt = os.path.join(path, network_name,'protos', 'faster_rcnn_test.pt')
        if network_name.find('Res')>=0:
            if not network_name.find('coco')>=0:
                self.prototxt = os.path.join(path,'rfcn', self.NETS[network_name][0],'test_agnostic.prototxt')
            else:
                self.prototxt = os.path.join(path,'rfcn', 'coco',self.NETS[network_name][0],'test_agnostic.prototxt')

        self.net = None
        self.unload_net_timer = None
        self.gpu_id = gpu_id
        self.service_queue = 0


        self.caffemodel = os.path.join(path, 'caffe', self.NETS[network_name][1])

        if not os.path.isfile(self.caffemodel):
            raise IOError(('{:s} not found.\nDid you run ./data/script/'
                       'fetch_faster_rcnn_models.sh?').format(self.caffemodel))

        s = rospy.Service('deep_object_detection/detect_objects', DetectObjects, self.handle_detect_objects_req)
        s2 = rospy.Service('deep_object_detection/get_labels', GetLabels, self.handle_getlabels_req)
        self.detectedobjectspub = rospy.Publisher("deep_object_detection/detected_objects",DetectedObjects,queue_size=10)
        rospy.loginfo("Ready to detect objects")
        rospy.spin()
