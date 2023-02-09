# Overview
The aim of this project is to explore different approaches to the problem of multiple object tracking through the concept of tracking-by-detection. Different detectors have been implemented (IoU-based, ReID-based, GNN-based) and different track processing techniques have been applied.
The project is based on lectures and several project assignments from the CV3DST course at the Technical University of Munich.

# Data 
1. The Market dataset was used to train the Reid model
2. The MOT16 dataset was used to train, evaluate and test trackers

# Trackers
Various approaches to object tracking have been researched and implemented. The corresponding notebooks contain the motivation behind different tracking approaches, mostly taken from the cv3dst course and its project assignments. As well as the results of the detector and visualisations of the predictions.

1) [IoUTracker](notebooks/2.5-iou-tracker.ipynb ) - 
2) [HungarianIoUTracker](notebooks/3.0-hungarian_algo.ipynb) - implementation of Hungarian algorithm for bbox matching improvement
3) [ReIDHungarianIoUTracker](notebooks/5.0-appearance-tracker.ipynb) - appearance net(ReID) + IoU(bbox) + hungarian matching 
4) [LongTermReIDHungarianTracker](notebooks/6.0-mg-LongTermReidTracker.ipynb) - added the ability to restore the track even if the object has been out of frame for a while
5) [MPNTracker](notebooks/7.0-gnn.ipynb) - Message Passing Network Tracker - a tracker is implemented in which we refine the quality of matching detections with tracks using a graph neural network (based on the message passing framework)

## ReID models
Several models on the market have been trained for use in trackers, and several interesting features have been implemented from scratch.
- triplet loss / cosine loss
- Hard Negative mining  
[example ipynb](notebooks/4.0-reid-net.ipynb)

## Graph neural network
GNN implementation based on  message passing framework
[explanation ipynb](notebooks/7.0-gnn.ipynb)


# Results 

## gif with the results of work on the sequences MOT16-02 and MOT16-11

![MOT16-02 tracker results ](/output/figs/MOT16-02-result.gif) ![MOT16-11 tracker results](/output/figs/MOT16-11-result.gif) 

## Table results concentrate on IDF1 and MOTA

|          | IDF1     |            |                  |               |       |   | MOTA     |            |                  |               |       |   |
|----------|----------|------------|------------------|---------------|-------|---|----------|------------|------------------|---------------|-------|---|
|          | Baseline | Hungarian  | ReidHungarianIoU | Long-TermReID | GNN   |   | Baseline | Hungarian  | ReidHungarianIoU | Long-TermReID | GNN   |   |
| MOT16-02 | 32.2%    | 39.1%      | 38.8%            | 45.9%         | **48.5%** |   | 13.5%    | 48.9%      | 48.8%            | 49.4%         | **49.6%** |   |
| MOT16-11 | 49.0%    | 60.4%      | 62.8%            | 68.3%         | **70.3%** |   | 28.7%    | 76.3%      | 76.5%            | 75.9%         | **77.0%** |   |
| OVERALL  | 40.6%    | 49.75      | 50.8             | 57.1          | **59.4%** |   | 21.1%    | 62.6%      | 62.65%           | 62.65%        | **63.3%** |   |



## Folder Structure

    .
    ├── data
    ├── models               # trained models
    ├── notebooks            # ipynb notebooks
    ├── output               # folder for files with tracker results
    ├── src  
    │   |── market           # folder with modules for working with market dataset and learning reid model
    │   |── mot
    │   |   ├──data           # mot16 dataset datasets and dataloaders
    │   |   ├──model          # torch modules: gnn module, object  detection module, reid model
    │   |   └──tracker        # several tracker implementations
    │   │       
    │   ├── __init__.py
    │   ├── eval.py
    │   ├── trainer.py
    │   ├── transforms.py
    │   ├── utils.py 
    │   └── visualize.py  
    │               
    ├── .deepsource.toml
    ├── .gitattributes
    ├── .gitignore
    ├── README.md

## Usage

1. Clone the repo
   ```sh
   git clone https://github.com/Kristobal-Khunta/multi-object-tracking.git
   git lfs
   ```
2. download datasets and store it in ./data
- [MOT16 dataset mainpage](https://motchallenge.net/data/MOT16/)
- [market dataset mainpage](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html)
3. Install dependencies
   ```sh
   pip install -r requirements.txt
   ```
4. eval tracker with prefefined weighs
    ```sh
   python -m eval_tracker
   ```
## optional:
- train reid net on market data
   ```sh
   python -m train_reid
   ```
- train gnn net on crops
    ```sh
   python -m train_gnn
   ```