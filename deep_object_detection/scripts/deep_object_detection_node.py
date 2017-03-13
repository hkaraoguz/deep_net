#!/usr/bin/env python

import rospy
import argparse

from deep_object_detection.deep_object_detection import *


if __name__=="__main__":

  """Parse input arguments."""

  parser = argparse.ArgumentParser(description='Faster R-CNN Object Detection')

  parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use. Default -1 (CPU Mode Enabled)',
                      default=-1, type=int)

  parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16,vgg16KTH,zf,ResNet-50,ResNet-101,ResNet-101coco]', default='vgg16')

  args = parser.parse_args((rospy.myargv()[1:]))

  rospy.init_node("deep_object_detection_node")
  rospy.loginfo("Starting deep object detection node")

  FasterRCNNPascal(args.gpu_id,args.demo_net)
