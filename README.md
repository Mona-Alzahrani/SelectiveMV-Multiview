# 3D Object Classification With Selective Multi-View Fusion And Shape Rendering
This repository is for the following paper _"3D Object Classification With Selective Multi-View Fusion And Shape Rendering"_ introduced by [Mona Alzahrani](https://github.com/Mona-Alzahrani), Muhammad Usman, Randah Alharbi, [Saeed Anwar](https://saeed-anwar.github.io/), Ajmal Mian, and Tarek Helmy, 2024.

## Requirements: 
The model is built in _Visual Studio Code_ editor using: 
* Python 3.9.16
* Tensorflow-gpu 2.7
* pytorch 2.0.1
* Pytorch-cuda 11.7
* keras 2.6
* Transformers 4.38.2
  

## Content:
1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Dataset](#dataset)
4. [Getting Started](#getting-started)
5. [Training](#training)
6. [Testing](#testing)
7. [Results](#results)

## Introduction:
3D classification is complex and challenging because of high-dimensional data, the intricate nature of their spatial relationships, and viewpoint variations. We fill the gap in view-based 3D object classification by examining the factors that influence classification's effectiveness via determining their respective merits in feature extraction for 3D object recognition by comparing CNN-based and Transformer-based backbone networks side-by-side. Our research extends to evaluating various fusion strategies to determine the most effective method for integrating multiple views and ascertain the optimal number of views that balances classification and computation. We also probe into the effectiveness of different feature types from rendering techniques in accurately depicting 3D objects. This investigation is supported by an extensive experimental framework, incorporating a diverse set of 3D objects from the ModelNet40 dataset. Finally, based on the analysis, we present a Selective Multi-View deep model (SelectiveMV) that shows efficient performance and provides high accuracy given a few views.
  

## Architecture:
The architecture of the proposed Selective Multi-View deep model (SelectiveMV) consists of five phases::
 <br /> (A) **View rendering:** multiple m views are captured from different viewpoints of a given 3D object. Based on the rendering technique, the rendered views are either grayscale, shaded or depth maps. 
 <br /> (B) **Feature extraction:** each extracted view is input into a pre-trained backbone network to extract the corresponding feature sets.
 <br /> (C) **Vectorization:** the detected feature sets are flattened into vectors.
 <br /> (D) **Selective fusion:** the feature vectors are compared based on their similarity using Cosine Similarity, and a vital score is obtained and normalized. The views with higher scores are selected and fused using a fusion technique to generate a global descriptor.
 <br /> (D) **Classification:** the global descriptor of the object is fed into a classifier to predict its class. The boxes in blue and the number of selected views are the key variables impacting the classification performance we evaluate.

<p align="center">
  <img align="center"  src="/images/Methodology3.png" title="Illustration of the proposed framework">
  <figcaption>The architecture of proposed Selective Multi-View deep model (SelectiveMV). The blue boxes (in addition to number of selected views) are the key variables that we evaluate their impact in the classification performance.</figcaption>
</p>


## Dataset:
All experiments in this study utilize the widely recognized [**ModelNet40**](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Wu_3D_ShapeNets_A_2015_CVPR_paper.html). The dataset comprises 12,311 grayscale objects categorized into 40 classes, with standard training and test splits. Specifically, it includes 9,843 objects for training and 2,468 objects for testing. It is important to note that the number of objects varies across categories, resulting in an imbalanced distribution. Therefore, two metrics, Overall Accuracy (OA) and Average Accuracy (AA), are reported. 

In order to capture multiple views from each 3D object, a circular camera setup is employed, similar to [MVCNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.html). In many related studies, such as [MVCNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.html), [RotationNet](https://openaccess.thecvf.com/content_cvpr_2018/html/Kanezaki_RotationNet_Joint_Object_CVPR_2018_paper.html), [view-GCN](https://openaccess.thecvf.com/content_CVPR_2020/html/Wei_View-GCN_View-Based_Graph_Convolutional_Network_for_3D_Shape_Analysis_CVPR_2020_paper.html), and [MVTN](https://openaccess.thecvf.com/content/ICCV2021/html/Hamdi_MVTN_Multi-View_Transformation_Network_for_3D_Shape_Recognition_ICCV_2021_paper.html), 12 virtual cameras are positioned around the object, resulting in the extraction of 12 rendered views:

<p align="center">
  <img align="center"  src="/images/intro-2.gif" >
 </p> 

Furthermore, this study investigates different views with distinct feature types using various image rendering techniques. The following three representation views with varying types of features are explored:
 <p align="center">
  <img align="center"  src="/images/ShapeRepresentation3.png" >
 </p> 

* **Grayscale Views**: These views employ surface normal maps generated using the [Phong reflection](https://dl.acm.org/doi/abs/10.1145/280811.280980) model. The mesh polygons are rendered under a perspective projection, and the color of the shape in general will be grayscale as the original 3D object. While the Grayscale color of each pixel is determined by interpolating the reflected intensity of the polygon vertices. The shapes are uniformly scaled to fit within the viewing volume. Samples of Grayscale views are: 
  <p align="center">
    <img align="center" src="/images/grayscale1.png">
  </p>
  
  For a fair comparison, we used the same grayscale views rendered by [MVCNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.html):
  * **GreyscaleModelNet40v1 Training** can be download from [here.](https://drive.google.com/file/d/1ZTG6DkXhR0ee8tJAUkbPGncGL98t8LqS/view?usp=sharing)
  * **GreyscaleModelNet40v1 Testing** can be download from [here.](https://drive.google.com/file/d/1yrNSe9YghIXm9s0kJTuzJC5oZYhrVMOe/view?usp=sharing)

  
* **Shaded Views**: These views are also rendered using the Phong reflection model, but the resulting images are grayscale with a black background. The camera's field of view is adjusted to encapsulate the 3D object within the image canvas tightly. Samples of Shaded views are:
  <p align="center">
    <img align="center" src="/images/shaded1.png">
  </p>

  For a fair comparison, we used the same shaded views rendered by [MVCNN-new](https://openaccess.thecvf.com/content_eccv_2018_workshops/w18/html/Su_A_Deeper_Look_at_3D_Shape_Classifiers_ECCVW_2018_paper.html):
  * **ShadedModelNet40 Training** can be download from [here.](https://drive.google.com/file/d/1xxTCtDfTJDdEpkNQOoCvtGM6va2NYV2r/view?usp=sharing)
  * **ShadedModelNet40 Testing** can be download from [here.](https://drive.google.com/file/d/1WIuRJe7Oz0vVLi1fAsbELyfm95A09EdI/view?usp=sharing)


* **Depth Views**: In this case, the views solely record the depth value of each pixel. Samples of Depth views are
 <p align="center">
    <img align="center" src="/images/depth1.png">
  </p>
 

  For a fair comparison, we used the same depth views rendered by [MVCNN-new](https://openaccess.thecvf.com/content_eccv_2018_workshops/w18/html/Su_A_Deeper_Look_at_3D_Shape_Classifiers_ECCVW_2018_paper.html):
  * **DepthdModelNet40 Training** can be download from [here.](https://drive.google.com/file/d/1oikKSx8ksaepEq8kMLEB9-EIPCkiIS4Z/view?usp=sharing)
  * **DepthdModelNet40 Testing** can be download from [here.](https://drive.google.com/file/d/1NviSUPru3QmMgKzP08kQuzOk1zz9i6a1/view?usp=sharing)

## Getting Started:
Since we will experiment different rendering techniques, backbone networks, numbers of selected views, fusion startegies, and classifiers, we do the following to organize the data and results: 

* Prepare two folders:
   * data folder: put all the unzip dataset folders inside it. Note that each dataset will have two folders one for training and other for testing. The data folder will also be used later by our code to save the extracted features.
   * Results folder: create new folders inside it and rename them with dataset names (one folder for each dataset). 
        * And inside each dataset folder, create new folders and rename them with backbone network names (pre-trained CNN or transformer).
    
  The folders will look like this:
  ```
  ---------data
            --------modelnet40v1_train
            --------modelnet40v1_test
            --------shaded_modelnet40v1_train
            --------shaded_modelnet40v1_test
            --------depth_modelnet40v1_train
            --------depth_modelnet40v1_test

  ---------Results
            --------modelnet40v1
                      --------VGG16
                      --------VGG19
                      --------ResNet50
                      --------ResNet152
                      --------EfficientNetB0
                      --------ViT
                      --------BEiT
            --------shaded_modelnet40v1
                                .
                                .
            --------depth_modelnet40v1
                                .
                                .
  ```
   
## Feature Extraction:
For feature extraction, we used the following seven pre-trained backbones seperetly:
              <p align="center">
                <img align="center" src="/images/backbones.png">
             </p>

## Training and Testing:             
To run an experiment, use the following guidline to guide you to which files you should run for training and testing:
* **Single View Experiment**: run **Training-SV+Voting.ipynb** for training, and **Testing-SV.ipynb** for testing. Note: all samples are used for training; and no fusion needed in testing.
* **Majority-Voting Multi-view Experiment**: run **Training-SV+Voting.ipynb** for training, and **Testing-MV-Voting.ipynb** for testing. Note: all samples are used for training; and late Majority-Voting fusion needed in testing.
* **Max-pooling Multi-view Experiment**: run **Training-MV-Max+Sum.ipynb** for training, and **Testing-MV-Max+Sum.ipynb** for testing. Note: early Max-pooling fusion needed in training and testing.
* **Sum-pooling Multi-view Experiment**: run **Training-MV-Max+Sum.ipynb** for training, and **Testing-MV-Max+Sum.ipynb** for testing. Note: early Sax-pooling fusion needed in training and testing.

The following need to be specified before experiment running in all training and testing files:
1. Track and replace the paths of data and Results folders with your paths:
   ```shell
   "./Results/"
   "./data/"
   ```
2.   Choose the dataset version and path:
   ```shell
dataset_version= 'original_modelnet40v1'  
dataset_train = './data/original_modelnet40v1_train'
```
OR
```shell
dataset_version= 'shaded_modelnet40v1'    
dataset_train = './data/shaded_modelnet40v1_train' 
```
OR
```shell
dataset_version= 'depth_modelnet40v1' 
dataset_train = './data/depth_modelnet40v1_train'
```
3.   Spicify the img size (here 224*224)
```shell
Img_Size= 224
```
4.   Spicify the backbone feature extractor (here BEiT); and run its following code cells. Note: we only experimented seven backbones but more options are included in the code.
```shell
all_model_name_txt = ["BEiT"]
````
5.   Spicify the BATCH_SIZE, Mini_BATCH_SIZE, EPOCHS, learning_rate:
```shell
BATCH_SIZE = 384
Mini_BATCH_SIZE = 32
EPOCHS = 30
learning_rate = 0.0001 
```

## Results:
### Quantitative Results:
The classification accuracy results of the proposed models using the ModelNet40v1 and ModelNet40v2 datasets when the models trained for 30 epochs are summarized in the bellow Table. This table presents the outcomes of various experiments conducted under different settings. It is worth noting that the proposed approach achieves the best results, an Overall Accuracy (OA) of 83.63% and Average Accuracy (AA) of 83.63%, when only a single view is used for classifying 3D objects. This is observed when the pre-trained ResNet-152 model is employed for feature extraction, and the FCN is used as the classifier, trained with 12 views from ModelNet40v1 dataset (model M<sub>13</sub>). Additionally, when the same feature extractor is trained with 20 views from the ModelNet40v2 dataset, the proposed approach with the FCL classifier demonstrates competitive performance, achieving an OA of 83.7%, but with an AA of 80.39% (model M<sub>15</sub>).

The classification accuracy of our proposed model on ModelNet40v1 and ModelNet40v2 datasets is rendered as 12 views and 20 views for each object, respectively. Each model is trained for 30 epochs. The best results are shown in bold and underlined:
  <p align="center">
    <img align="center" src="/images/allResults.png">
  </p>

Aso, we investigate the effect of shape representation on the classification of a single view for rendering 3D objects. We utilized the ModelNet40v2 dataset for this experiment, with 12 views per 3D object. However, each 3D object was rendered using the [Phong shading technique](https://dl.acm.org/doi/pdf/10.1145/280811.280980). Shading techniques have been demonstrated to improve performance in models such as [MVDAN](https://link.springer.com/article/10.1007/s00521-021-06588-1) and [MVCNN](https://openaccess.thecvf.com/content_eccv_2018_workshops/w18/html/Su_A_Deeper_Look_at_3D_Shape_Classifiers_ECCVW_2018_paper.html). The rendered views were grayscale images with dimensions of 224*224 pixels and black backgrounds, as depicted in the bellow Figure. The camera's field of view was adjusted so that the image canvas tightly encapsulated the 3D object.

Results of the proposed model with shading as rendering technique:
  <p align="center">
    <img align="center" src="/images/shadedResults.png">
  </p>


Comparison with the selective view-based 3D object classification methods experimented with a single view. OA is overall accuracy, and AA is average accuracy. The best results are shown in bold and underlined:
  <p align="center">
    <img align="center" src="/images/ComparisonResults.png">
  </p>
 
### Visual Results:
In this work, we consider and experiment with the best discriminative view differently. The first selection technique, considers the _Most Similar View (MSV)_ as a considerably reasonable discriminating view because it could contain most features on other views corresponding to the same object. The _MSV_ has a higher cosine similarity (a higher important score). The other way is by considering the _Most Dissimilar View (MDV)_ as the best discriminative view due to the unique and irredundant features of different views corresponding to the same object. The _MDV_ is the view that has the lower cosine similarity (lower important score).
<p align="center">
    <img align="center" src="/images/MSV_and_MDV.png">
    <figcaption> The set of 12 circular views obtained from sample objects and their corresponding importance scores are displayed. Views with the highest importance scores, representing the Most Similar Views (MSV), are highlighted with green boxes. Conversely, views with the lowest importance scores, representing the Most Dissimilar Views (MDV), are enclosed in brown boxes.. </figcaption>
  </p>

Here, we use the [Grad-CAM](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html) technique to analyze the predicted labels to highlight the regions on the views responsible for the classification. We show some correctly predicted views by the proposed model with their corresponding feature maps highlighted with Guided GradCam showing the responsible regions that led to the correct classification. These feature maps show how the proposed model selects the views that contain distinguishing features, such as shelves in bookshelves and circular edges in bowls.
<p align="center">
    <img align="center" src="/images/CorrectClasses1.png">
    <figcaption>Samples of feature maps belong to correctly classified labels highlighted with the Grad-CAM technique to show the responsible regions that led to the classification. </figcaption>
  </p>

It has been found that top confusions happened when: i) "flower pot" predicted as "plant", ii) "dressers" predicted as "night stand", and iii) "plant" predicted as "flower pot". Even for human observers, distinguishing between these specific pairs of classes can be challenging due to the ambiguity present.
<p align="center">
    <img align="center" src="/images/MissclassifiedClasses.png">
    <figcaption>Multi-view samples from ModelNet40v1 dataset of the most wrongly classified objects by the proposed model. </figcaption>
  </p>


  Since _Most Similar Views (MSV)_ give better results, input and output of the proposed model will be as follow: Given a 3D object as input, our proposed model generates _m_ multi-view images from the 3D object and assigns importance scores based on their cosine similarity, in which the view with the highest importance score is selected as the global descriptor to classify the object and finally, predict its category as output.

<p align="center">
    <img align="center" src="/images/SelectedView.png">
    <figcaption>Input and output of the proposed model. </figcaption>
  </p>


## Citation:
For those who find the provided code beneficial for their research or work, we kindly request citing the following paper:
```
@article{SelectiveMV2024,
  title={Selective Multi-View Deep Model for 3D Object Classification},
  author={Alzahrani, Mona and Usman, Muhammad and Alharbi, Randah and Anwar, Saeed and Mian, Ajmal and Helmy, Tarek},
}
```

"Please note that the paper is forthcoming. Once the paper is officially published, we will update the citation details accordingly."

## Acknowledgement:
This project is funded by the Interdisciplinary Research Center for Intelligent Secure Systems at King Fahd University of Petroleum & Minerals (KFUPM) under Grant Number INSS2305.

