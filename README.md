

# Overview
The aim of this project is to explore different approaches to the problem of multiple object tracking through the concept of tracking-by-detection. Different detectors have been implemented (IoU-based, ReID-based, GNN-based) and different track processing techniques have been applied.
The project is based on lectures and several project assignments from the CV3DST course at the Technical University of Munich.

# data 
1. The Market dataset was used to train the Reid model
2. The MOT16 dataset was used to train, evaluate and test trackers

# code structure


# Trackers
Various approaches to object tracking have been researched and implemented. The corresponding notebooks contain the motivation behind different tracking approaches, mostly taken from the cv3dst course and its project assignments. As well as the results of the detector and visualisations of the predictions.

1) [IoUTracker](notebooks/2.5-iou-tracker.ipynb ) - 
2) [HungarianIoUTracker](notebooks/3.0-mg-hungarian_algo.ipynb)
3) [ReidTracker](notebooks/4.0-mg-reid-net.ipynb)
4) [LongTermReIDHungarianTracker](notebooks/6.0-mg-LongTermReidTracker.ipynb)
5) [MPNTracker](notebooks/8.0-mg-tracker-inference.ipynb) - with message passing network (graph neural networks) for prediction refinement

### reid models
Several models on the market have been trained for use in trackers, and several interesting features have been implemented from scratch.
- triplet loss / cosine loss
- Hard Negative mining 

### Graph neural network
GNN implementation based on neural message passing framework

# Results 
## GIF
![MOT16-02 tracker results ](/output/figs/MOT16-02-result.gif) ![MOT16-11 tracker results](/output/figs/MOT16-11-result.gif) 

## Table results concentrate on IDF1 and MOTA

|          | IDF1     |            |                  |               |       |   | MOTA     |            |                  |               |       |   |
|----------|----------|------------|------------------|---------------|-------|---|----------|------------|------------------|---------------|-------|---|
|          | Baseline | Hungarian  | ReidHungarianIoU | Long-TermReID | GNN   |   | Baseline | Hungarian  | ReidHungarianIoU | Long-TermReID | GNN   |   |
| MOT16-02 | 32.2%    | 39.1%      | 38.8%            | 45.9%         | **48.5%** |   | 13.5%    | 48.9%      | 48.8%            | 49.4%         | **49.6%** |   |
| MOT16-11 | 49.0%    | 60.4%      | 62.8%            | 68.3%         | **70.3%** |   | 28.7%    | 76.3%      | 76.5%            | 75.9%         | **77.0%** |   |
| OVERALL  | 40.6%    | 49.75      | 50.8             | 57.1          | **59.4%** |   | 21.1%    | 62.6%      | 62.65%           | 62.65%        | **63.3%** |   |

## How To Use

1. train reid net on market data
2. train gnn net on crops
3. use tracker

#  python -m eval_tracker --device='cuda:1' 
