# Image Captioning Project

This repository contains an image captioning project which uses deep learning models. Given an image, first, it is processed by a CNN, and second, by a RNN. The model outputs a text caption of the content. The [MS COCO](https://cocodataset.org/#home) dataset was used for training.

The implementation is based on the paper [*Show and Tell* by Vinyals et al.](https://arxiv.org/abs/1411.4555) and the project uses materials from the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891), available in their original form in the repository [CVND---Image-Captioning-Project](https://github.com/udacity/CVND---Image-Captioning-Project).


## Introduction

:construction: TBD

Table of Contents:

- [Image Captioning Project](#image-captioning-project)
  - [Introduction](#introduction)
  - [How to Use This](#how-to-use-this)
    - [Overview of Files and Contents](#overview-of-files-and-contents)
    - [Dependencies](#dependencies)
    - [COCO Dataset](#coco-dataset)
  - [Content Topic](#content-topic)
  - [Improvements, Next Steps](#improvements-next-steps)
  - [Interesting Links](#interesting-links)
  - [Authorship](#authorship)

## How to Use This

:construction: TBD

### Overview of Files and Contents

:construction: TBD

First, you need to install the [dependencies](#dependencies) and set up the [COCO dataset](#coco-dataset); then, you can open and run the notebooks. Note that these need to be executed sequentially; each of them performs the following tasks:

- `0_Dataset.ipynb`
  - The `cocoapi` for the [COCO dataset](#coco-dataset) is tested; the dataset is loaded and it is shown how to get images and captions with the API.
- `1_Preliminaries.ipynb`
  - The `DataLoader` class which is built using `data_loader.py` and `vocabulary.py` is explained; also the code in those auxiliary scripts is explained with examples.
  - The model definition is tested (size of the output).
- `2_Training.ipynb`
  - Hyperparameters are defined following [Vinyals et al.](https://arxiv.org/abs/1411.4555).
  - The model is trained.
- `3_Inference.ipynb`

- `data_loader.py`: data loader class based on the Pytorch [DataLoader](https://pytorch.org/docs/master/data.html#torch.utils.data.DataLoader) and the COCO API.
  - A dataset is built using the COCO API.
  - Images are loaded and returned as tensors.
  - Captions are processed to build a vocabulary (see next file); then, captions are returned as tensors of indices.
  - The data loader makes possible to yield batches of image-captions
  - We get the data loader via `get_loader()` and we can pass a `transform` to it
  - Once we've found a correct `vocab_threshold` (see below), we should use the option `vocab_from_file=True`, because the persisted vocabulary is loaded.
  - Note: the index we insert to the COCO dataset is not the image index, but the annotation index; then, the image id of that annotation is found, and with it the image. Thus, we end up having the same image every 5 captions. However, not that the caption ids of the same image don't need not be consecutive!
- `vocabulary.py`: vocabulary class based on NLTK.
  - All captions are read and tokenized with NLTK.
  - A vocabulary word is created (with associated index) if the word appears more than `vocab_threshold` times.
  - The vocabulary is stored in a dictionary: `word2idx`.
  - The vocabulary object is a callable that returns the index of a token/word in it.
  - We have the special tokens `<star>` (index 0), `<end>` (index 1) and `<unk>` (index 2); all captions are wrapped with 0/1 indices and unknown words have the index 2.
  - Building the vocabulary takes some minutes.
  - The built vocabulary is persisted as a pickle.
  - Once we've found a correct `vocab_threshold`, we should use the option `vocab_from_file=True`, because the persisted vocabulary is loaded.

- `model.py`: definition of the `EncoderCNN` and the `DecoderRNN`
  - `EncoderCNN`: frozen ResNet50 from which its classifier is replace by a new fully connected layer that maps feature vectors into vectors of the size of the word embedding.
  - `DecoderRNN`: architecture based in the one from [Vinyals et al.](https://arxiv.org/abs/1411.4555). It consists in an LSTM layer which takes the caption sequence with the transformed image at the front. The output is a sequence of hidden states of the same length; the hidden states are mapped to the vocabulary space so that each sequence element predicts the likelihood of any word in the vocabulary.



### Dependencies

You should create a python environment (e.g., with [conda](https://docs.conda.io/en/latest/)) and install the dependencies listed in the [requirements.txt](requirements.txt) file.

A short summary of commands required to have all in place is the following:

```bash
conda create -n img-capt python=3.6
conda activate img-capt
conda install pytorch torchvision -c pytorch
conda install -c anaconda scikit-image
conda install pip
pip install -r requirements.txt
```

### COCO Dataset

```bash
# Clone the COCO-API package and build it
cd ~/git_repositories
git clone https://github.com/cocodataset/cocoapi.git  
cd cocoapi/PythonAPI
conda activate img-capt
make 
cd ..
# IMPORTANT:
# You might need to include this path/line to your code:
# sys.path.append('~/git_repositories/cocoapi/PythonAPI')
#
# Now, download the dataset
# and place it in the cocoapi repo folder. 
# Follow the instructions below!
```

The COCO dataset can be downloaded from [https://cocodataset.org/#download](https://cocodataset.org/#download). I used all the images and annotations of the 2014 dataset, which are linked below:

- [Train 2014 Images](http://images.cocodataset.org/zips/train2014.zip)
- [Valid 2014 Images](http://images.cocodataset.org/zips/val2014.zip)
- [Test 2014 Images](http://images.cocodataset.org/zips/test2014.zip)
- [Train/Valid 2014 Annotations](http://images.cocodataset.org/annotations/annotations_trainval2014.zip)
- [Test 2014 Annotations](http://images.cocodataset.org/annotations/image_info_test2014.zip)

Note that the dataset weights more than 20 GB.

The images and annotations need to be extracted to the `cocoapi` repository folder we created, to the folders `cocoapi/annotations` and `cocoapi/images`. The final structure is the following:

```
.
├── PythonAPI
├── annotations
│   ├── captions_train2014.json
│   ├── captions_val2014.json
│   ├── image_info_test2014.json
│   ├── instances_train2014.json
│   ├── instances_val2014.json
│   ├── person_keypoints_train2014.json
│   └── person_keypoints_val2014.json
├── images
│   ├── train2014/
│   │   ├── ...
│   ├── val2014/
│   │   ├── ...
│   └── test2014/
│   │   ├── ...
├── ...
```

## Content Topic

:construction: TBD

[text_sentiment](https://github.com/mxagar/text_sentiment)

- Two different models are defined but optimized together since their selected parameters are passed to the optimizer.

## Improvements, Next Steps

:construction: TBD

- Encoder: try other frozen/pre-trained backbones (e.g., [Inception-V3](https://pytorch.org/hub/pytorch_vision_inception_v3/)) as feature extractors.
- Encoder: add batch normalization after the feature extractor.
- Implement attention, for instance after [Show, Attend and Tell, by Xue et al.](https://arxiv.org/abs/1502.03044)
- Implement *beam search* to sample the tokens of predicted sentence, as in [Show and Tell, by Vinyals et al.](https://arxiv.org/abs/1411.4555)

## Interesting Links

- [My notes and code](https://github.com/mxagar/computer_vision_udacity) on the [Udacity Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891).
- [My notes and code](https://github.com/mxagar/deep_learning_udacity) on the [Udacity Deep Learning Nanodegree](https://www.udacity.com/course/deep-learning-nanodegree--nd101).
- [Pytorch: Writing Custom Datasets, DataLoaders and Transforms](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [A Tutorial on Image Captioning](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)

**Papers** -- look in the folder [literature](literature/literature.txt):

- [Vinyals et al.: Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
- [Xu et al.: Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/abs/1502.03044)

## Authorship

Mikel Sagardia, 2022.  
No guarantees.

You are free to use this project, but please link it back to the original source.