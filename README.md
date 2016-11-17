# Deep Net Package

Deep Net package employs both [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) and [py-R-FCN](https://github.com/Orpine/py-R-FCN) as a ROS package for object detection. It can detect the 20 object categories of the Pascal object detection challenge. Moreover, it contains a fine-tuned model for detecting *chairs, laptops and monitors*.

Prerequisites for GPU version (py-faster-rcnn)
---
* You need to have CUDA installed. The package is tested with CUDA 7.0 and 7.5.
* You need to download the CUDNN V3. After downloading and extracting you should add the path of the *include* and *lib64* folders to your library and include paths:
```
export CPATH=/path_to_cudnnv3/include:$CPATH
export LIBRARY_PATH=/path_to_cudnnv3/lib64:$LIBRARY_PATH
```
* Clone the py-faster-rcnn [repo](https://github.com/rbgirshick/py-faster-rcnn). We will call it *rcnn-repo*
* Navigate into the repo directory. Under the repo directory, you need to clone the specific version of the Caffe. Use the following commands to get the required version:
```
git clone https://github.com/rbgirshick/caffe-fast-rcnn.git
git checkout 0dcd397b29507b8314e252e850518c5695efbb83 .
```
* After cloning the Caffe repo, modify the *Makefile.config* file. Uncomment the *USE_CUDNN* flag and set it to 1.
Also uncomment the *WITH_PYTHON_LAYER* flag and set it to 1. Build the Caffe libraries. Do not forget to make the python libraries also using the command `make pycaffe`.

* After Caffe is built, navigate to the *rcnn-repo/lib* and run make. *Hint:* If you have an older GPU such as 6XX series, then in the `setup.py` file, change the `-arch=sm_35` option to `-arch=sm_30` and run make.

* You need to add the library and include paths of this specific build to the regular search paths. Use the following commands to do this:
```
export LD_LIBRARY_PATH=/rcnn-repo/lib:/rcnn-repo/caffe-fast-rcnn/build/lib:
/path_to_cudnnv3/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/rcnn-repo/lib:/rcnn-repo/caffe-fast-rcnn/python:$PYTHONPATH
```

Prerequisites for GPU version (py-R-FCN)
---
* You need to have CUDA installed. The package is tested with CUDA 7.5.
* You need to download the CUDNN V5. After downloading and extracting you should add the path of the *include* and *lib64* folders to your library and include paths:
```
export CPATH=/path_to_cudnnv5/include:$CPATH
export LIBRARY_PATH=/path_to_cudnnv5/lib64:$LIBRARY_PATH
```
* Clone the py-R-FCN [repo](https://github.com/Orpine/py-R-FCN). We will call it *rfcn-repo*
* Navigate into the repo directory. Under the repo directory, you need to clone the specific version of the Caffe. Use the following commands to get the required version:
```
git clone https://github.com/Microsoft/caffe.git
git checkout 1a2be8ecf9ba318d516d79187845e90ac6e73197
```
* After cloning the Caffe repo, modify the *Makefile.config* file. Uncomment the *USE_CUDNN* flag and set it to 1.
Also uncomment the *WITH_PYTHON_LAYER* flag and set it to 1. Build the Caffe libraries. Do not forget to make the python libraries also using the command `make pycaffe`.

* After Caffe is built, navigate to the *rfcn-repo/lib* and run make. *Hint:* If you have an older GPU such as 6XX series, then in the `setup.py` file, change the `-arch=sm_35` option to `-arch=sm_30` and run make.

* You need to add the library and include paths of this specific build to the regular search paths. Use the following commands to do this:
```
export LD_LIBRARY_PATH=/rfcn-repo/lib:/rfcn-repo/caffe/build/lib:
/path_to_cudnnv5/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/rfcn-repo/lib:/rfcn-repo/caffe/python:$PYTHONPATH
```

Usage
---
* Use the *download_models.sh* script, to download the Caffe model files.

* You can run the node using the command:
```
rosrun deep_object_detection object_detection_node.py
```
By default the node will run in CPU mode. In order to run on the GPU, you should set the gpu flag as `--gpu GPU_ID` where *GPU_ID* is the id of the GPU that you want to use which is usually 0.
* By default `VGG16` network will be used but you can also use the `ZF` network using `--net zf` or the fine-tuned 3-class KTH model using `--net vgg16KTH`.
* For using the RFCN models, you should use `--net ResNet-50` or `--net ResNet-101` commands.

## Services
There are 2 services advertised by this node:
* `/deep_object_detection/get_labels` This service will return you the list of object labels that can be recognized. There are 20 labels.
* `/deep_object_detection/detect_objects` This service will return the detected objects as a vector of images. The bounding box, label and confidence information of each detected object is returned.
