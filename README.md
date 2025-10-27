# signatureDetection
Handwritten signature detection using CV

## Dataset
For the best result the dataset used in the training is composed of images from pre-existing datasets and data on wich the model will be used on in practice.  
Every image is labelled with the bounding box of the signature, if one is present.  
To reduce the bias half of the examples present a signature, and half do not.  

Toy datasets:
- custom: for binary classification based on the presence of a signature, binary label
- custom_labelled: for object detection, with bounding boxes
- custom_yolo: dataset in YOLO fromat
- custom_dataset: dataset in COCO format: bbx, bby, bbw, bbh

- Tobacco: inverted images of documents, many of them with a signature
- NIST: images of tax documents, only one per doc. has a signature, every other has been discarded.
        the bb are all in the same position of the page -> data augmentation is needed to remove this bias

## Models
- Base classifier:
    a very simple classifer obtained from a pretrained CNN  
    very low accuracy (60%)  
    fast to train 
- FastRCNN detector:
    a detector from FastRCNN / RetinaNet
    using both images with and without examples leads to the model never making predictions and still getting rewarded half of the times
    => only use labelled data 
- YOLO detector:
    a detector from YOLOv8
    the dataset is too small for the model, no useful labels are outputted
    
## TODO
- add early stopping 
- encode the dataset for a faster loading