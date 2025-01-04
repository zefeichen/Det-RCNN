# Installation

1. Clone and enter this repository:
```
git clone git@github.com:zefeichen/Det-RCNN.git
cd Det-RCNN
```
2. Config Enviornmnet:

The configuration of the environment refers to trackformer, and the specific configuration process is shown on the following webpage
```commandline
https://blog.csdn.net/u010948546/article/details/131882768?spm=1001.2014.3001.5502
```

3. Download and unpack data and model
```
https://drive.google.com/drive/folders/1-5i5SLpV1NO2F2VbwHrub0U9rUOHe-D3?usp=share_link
```

4. The final folder structure should resemble this:
~~~
Det RCNN
|-- DataSets
    |-- CrowdHuman
    |   |-- annotation
    |   |   |-- annotation_train.odgt
    |   |   |-- annotation_val.odgt
    |   |   |-- train.json
    |   |   |-- val.json
    |   |-- body_head_annotations
    |   |   |-- instances_train_full_bhf_new.json
    |   |   |-- instances_val_full_bhf_new.json
    |   |-- Images
    |   |   |-- train
    |   |   |   |-- *.jpg
    |   |   |-- val
    |   |   |   |-- *.jpg
    |   |   |-- test
    |   |   |   |-- *.jpg
|-- weights
    |-- CrowdHuman.pth
~~~

5. train & val & inference
 ```
# train
python train.py

# val
python evaluate.py

# inference
python inference.py
```
