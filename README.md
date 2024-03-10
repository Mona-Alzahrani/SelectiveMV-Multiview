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
5. [Feature Extraction](#feature-extraction)
6. [Training and Testing](#training-and-testing)
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
                <img align="center" src="/images/backbones.png" width="600">
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
We start a comparison with existing 3D object classification models. Then, we perform a series of experiments to investigate the variables affecting the classification performance. Then, based on the mentioned analysis, we present the best-selected variables for SelectiveMV and analyze its predicted classes, followed by visualization and analysis of the fundamental selection mechanism. 
### Comparison with 3D Classification Models:
Our SelectiveMV model is benchmarked against existing state-of-the-art techniques within both view-based and model-based categories. SelectiveMV demonstrates exceptional performance, outperforming the alternatives, which is a testament to its robustness and effective design. It adeptly handles the input data's intricacies, whether in the form of multi-angle views or complex 3D models, further establishing its superiority in the current landscape of 3D object classification methodologies:
  <p align="center">
    <img align="center" src="/images/comparisionToRW.png" width="600">
  </p>
  
### Ablation Study:
We analyze the effect of backbone networks, rendering techniques, fusion strategy, classifiers, and the number of selected views. 
#### The Effect of the Backbone Network:
This experiment focuses on the accuracy of different backbone architectures and rendering techniques of SelectiveMV for feature extraction. These models were tasked with processing all the views, combined using max-pooling as a fusion strategy to generate the global descriptors which later classsify using FCL. The detailed results are reported in the following table:
  <p align="center">
    <img align="center" src="/images/BackboneResults.png" width="600">
  </p>
In general, ResNet-152 and BEiT-B stood out from the crowd, leading the charge as the most effective CNN-based and Transformer-based models, respectively, among all the rendering techniques. Delving into specifics, ResNet-152 showed impressive performance, especially when fed with shaded views, where it scored an OA of 91.82%. This made it the most proficient among its CNN-based peers. Where VGG-16, followed by VGG-19, were the worst-performing CNN backbones for all the rendering views. On the flip side, BEiT-B showed an OA of 90.72% . This performance edged out the ViT-B. Interestingly, these top models maintained high performance across all rendering techniques we considered, including grayscale, shaded, and depth. Given ResNet-152 and BEiT-B's standout performance, we will concentrate our efforts on them in subsequent experiments, as they have proven to be the most effective models among those tested. 

#### The Effect of the Rendering Technique:
From the above table, we have seen that the shaded technique was superior, followed by depth, then grayscale in all backbone networks. This fact applied even with the powerful ResNet-152 and BEiT-B. Except with the ViT-B backbone, depth results outperform others. Shaded views may be better because they can convey a more comprehensive set of visual information that aids neural networks in learning to recognize and classify objects more accurately. Depth views also provide valuable spatial information but may lack some surface detail, while grayscale views might omit important color information that could be crucial for distinguishing between similar objects. In the context of 3D object recognition, several works used the shaded technique as the only rendering technique to experiment with their proposed models, such as [MVA-CNN](https://link.springer.com/article/10.1007/s11042-019-7521-8), [MVDAN](https://link.springer.com/article/10.1007/s00521-021-06588-1), and [MVCVT](https://www.sciencedirect.com/science/article/abs/pii/S1047320323001566). However, it's important to note that the best rendering technique can be context-dependent.

#### The Effect of the Fusion Strategy:
The following table details the classification accuracies achieved by ResNet-152 and BEiT-B when subjected to various fusion methods with shaded rendering. Max-pooling emerged as a highly effective strategy for both architectures, although majority voting displayed competitive accuracies, mainly when a smaller number of views were utilized. For BEiT-B, the application of majority-voting led to an increase in performance, with OA reaching 92.54% and 92.79% upon fusing 3 and 6 views, respectively. In the case of ResNet-152, max-pooling generally yielded the highest accuracies across different view counts. However, an exception was noted with 12 views, where majority voting slightly improved the OA from 91.82% with max-pooling to 91.94%. Conversely, the sum-pooling technique resulted in a marginal decrease in classification performance for both neural network backbones. 
<p align="center">
    <img align="center" src="/images/FusionResults.png" width="600">
  </p>

#### The Effect of the Classifier:
Analyzing the classifiers results in the above table, it's clear that ResNet-152 and BEiT-B demonstrate varying degrees of compatibility with different classifiers. BEiT-B consistently outperforms with FCN when the majority voting, the best-performed fusion strategy, is utilized. BEiT-B favors pairing with an FCN, indicating that the FCN's more elaborate structure is beneficial for interpreting the consensus-based features derived from BEiT-B to predict classes for majority voting. However, with max-pooling, BEiT-B outperforms with FCL. The distinction underscores BEiT-B's flexible adaptability to different fusion approaches, optimizing its classification prowess with the appropriate combination of fusion strategy and classifier architecture.

On the other hand, ResNet-152 prefers FCL with max-pooling, the best-performed fusion strategy, when working with 6 or 12 views but switches its allegiance to FCN when the view count is reduced to 3 or just a single view. These insights suggest that while ResNet-152 may have a strong capacity for feature extraction, the optimal pairing with a classifier depends on the amount of viewpoint data available. With fewer views, the FCN can utilize the deep, complex features provided by ResNet-152, but with more views, the straightforward of an FCL may be more appropriate for achieving high accuracy.

However, BEiT-B can be considered better than ResNet-152 and all other backbones since it has the highest OA of 92.02% with just 3 views in shaded settings. It even outperformed other models with a single view, reaching an OA of 91.98%. This efficiency with minimal views arguably places BEiT-B at the top of the leaderboard, surpassing ResNet-152 and all other models we tested.

#### The Effect of the Number of Views:

In this experiment, each experimented rendering technique is considered separately with different quantities of selected views. This approach allows for an in-depth analysis of how the number of perspectives within each rendering technique impacts the classification performance of the 3D objects. The relationship between the number of views and the classification accuracy of various 3D classification models including our SelectiveMV model with BEit-B backbone, [Ma et al.](https://ieeexplore.ieee.org/abstract/document/8490588/), [Pairwise](https://openaccess.thecvf.com/content_cvpr_2016/html/Johns_Pairwise_Decomposition_of_CVPR_2016_paper.html), [MVCNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.html), and [3DShapeNets](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Wu_3D_ShapeNets_A_2015_CVPR_paper.html) is illustrated in the following figure:
<p align="center">
    <img align="center" src="/images/NoV3.png">
    <figcaption>Effect of the selected number of views. The number of views vs. overall accuracy of different 3D classification models, including our SelectiveMV, is plotted. G, Sh, and D refer to grayscale, shaded, and depth, respectively. </figcaption>
  </p>

Based on the analysis of the data presented, it can be observed that the performance of SelectiveMV models with a selected number of views depends on the rendering techniques. SelectiveMV (Shaded) presents the highest OA with 92.79%, suggesting its strategy is particularly effective with a limited number of views, such as 6, while OA drops to 90.88% when all the views are selected. These findings highlight the effectiveness of the proposed model in capturing essential features and suggest that even a smaller number of carefully chosen views can yield comparable results to a more extensive set of views. SelectiveMV (Depth) follows the same pattern with the number of views and an OA of 90.36% with 6 views, indicating it also handles a few view scenarios well, but with slightly less efficiency than SelectiveMV (Shaded). On the other hand, SelectiveMV (Grayscale) has the lowest OA among them and achieved its highest performance at 89.25\% when all the views were utilized, which might imply that its approach is less suited to fewer-view situations or that it requires more views to leverage its full potential. For those interested in the performance of the other backbones, with varying rendering techniques and number of views including single view, we have included those details in the supplementary material.


The comparative analysis illustrated above, clearly underscores the exceptional performance of our SelectiveMV (Shaded) model in the realm of 3D object classification. The model consistently outperforms established benchmarks such as [Ma et al.](https://ieeexplore.ieee.org/abstract/document/8490588/), [Pairwise](https://openaccess.thecvf.com/content_cvpr_2016/html/Johns_Pairwise_Decomposition_of_CVPR_2016_paper.html), [MVCNN](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Su_Multi-View_Convolutional_Neural_ICCV_2015_paper.html), and [3DShapeNets](https://www.cv-foundation.org/openaccess/content_cvpr_2015/html/Wu_3D_ShapeNets_A_2015_CVPR_paper.html) as evidenced by higher classification accuracy across varying numbers of views. Notably, SelectiveMV (Shaded) maintains its lead in accuracy irrespective of whether the input consists of fewer or more views. The only exception arises with the comparison to  [Ma et al.](https://ieeexplore.ieee.org/abstract/document/8490588/) model when utilizing 12 views; in this case, their model slightly edges out ours. Intriguingly, however, our SelectiveMV (Shaded) with just a single view can surpass the accuracy of Ma et al.'s 12-view model. This remarkable capability of SelectiveMV (Shaded) attests to its robustness and the sophistication of its approach, particularly when harnessing Shaded views. The implications of these findings are significant, as they not only validate the efficacy of our model but also position it as a frontrunner in 3D object classification.

 
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
@article{SelectiveMV2-2024,
  title={3D Object Classification With Selective Multi-View Fusion And Shape Rendering},
  author={Alzahrani, Mona and Usman, Muhammad and Alharbi, Randah and Anwar, Saeed and Mian, Ajmal and Helmy, Tarek},
}
```

"Please note that the paper is forthcoming. Once the paper is officially published, we will update the citation details accordingly."

## Acknowledgement:
This project is funded by the Interdisciplinary Research Center for Intelligent Secure Systems at King Fahd University of Petroleum & Minerals (KFUPM) under Grant Number INSS2305.

