Todo:
1. readme pretify
2. link to data download
3. gif
4. .py scripts final modules
5. train.py and infer.py скрипты - генерируют gif


## 
detector делает bbox  вокруг людей, трекер сопоставляет ббоксы на i-м кадре с уже отслеживающим треком объектом
## 
Это базовый 
Для задачи mot рассмотреть рассмотреть разные постепенно усложняющиеся подходы 
- от самых простых основанных на iou between bboxes до более сложных основанных на reid models и графовых моделях. 
Основанно на кодовой базе и домашнем задании от cv3dst

1) [IoUTracker](notebooks/2.5-iou-tracker.ipynb )
2) [HungarianIoUTracker](notebooks/3.0-mg-hungarian_algo.ipynb)
3) [ReidTracker](notebooks/4.0-mg-reid-net.ipynb)
4) [LongTermReIDHungarianTracker](notebooks/6.0-mg-LongTermReidTracker.ipynb)
5) [MPNTracker](notebooks/8.0-mg-tracker-inference.ipynb) - with message passing network (graph neural nets) для целей уточнения предсказаний

На датасете market обучены несколько reid моделей с
- triplet loss / cosine loss
- hard negative mining

# Results 
## GIF
![MOT16-02 tracker results ](/output/figs/MOT16-02-result.gif) ![MOT16-11 tracker results](/output/figs/MOT16-11-result.gif) 

## Table results

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
