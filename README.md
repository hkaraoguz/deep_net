# Deep Net Package

Deep Net package employs [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) deep neural network models as a ROS package for object detection. It can detect the 20 object categories of Pascal object detection challenge. Moreover, it contains a fine-tuned model for detecting *chairs, laptops and monitors*.

Prerequisites for GPU version
---
* You need to have CUDA installed. The package is tested with CUDA 7.0 and 7.5.
* You need to download the CUDNN V3. After downloading and extracting you should copy the contents of *lib64* and *include* folders of CUDNN to the local CUDA *lib64* and *include* folders.
* You need to have the specific version of Caffe repository. You can use the following commands to get the required version:
```
git clone https://github.com/rbgirshick/caffe-fast-rcnn.git
git checkout 0dcd397b29507b8314e252e850518c5695efbb83 .
```
After cloning the Caffe repository, modify the *Makefile.config* file. Uncomment the *USE_CUDNN* flag and set it to 1.
Also uncomment the *WITH_PYTHON_LAYER* flag and set it to 1. Build the Caffe libraries. Do not forget to make the python libraries also using the command `make pycaffe`.
After Caffe is built, append the library path of the Caffe to your existing library path:
```
export LD_LIBRARY_PATH=/path_to_caffe-fast-rcnn/build/lib:$LD_LIBRARY_PATH
```
* Append the python path of Caffe to the existing Python path:
```
export PYTHONPATH=/path_to_caffe-fast-rcnn/caffe-fast-rcnn/python:$PYTHONPATH
```
* Using the *download_faster_rcnn_models.sh* script, download the Caffe model files.
* Go to the *libpyfaster* directory under the *src* directory of the package and run make. *Hint:* If you have an older GPU such as 6XX series, then in the `setup.py` file, change the `-arch=sm_35` option to `-arch=sm_30` and run make.

Usage
---
You can run the node using the command:
```
rosrun deep_object_detection object_detection_node.py
```
By default the node will run in CPU mode. In order to run on the GPU, you should set the gpu flag as `--gpu GPU_ID` where *GPU_ID* is the id of the GPU that you want to use which is usually 0.
By default `VGG16` network will be used but you can also use the `ZF` network using `--net zf` or the fine-tuned 3-class KTH model using `--net vgg16KTH`.

## Services
There are 2 services advertised by this node:
* `/deep_object_detection/get_labels` This service will return you the list of object labels that can be recognized. There are 20 labels.
* `/deep_object_detection/detect_objects` This service will return the detected objects as a vector of images. The bounding box, label and confidence information of each detected object is returned.
