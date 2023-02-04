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

1) IoUTracker
2) HungarianIoUTracker
3) ReidTracker
4) LongTermReIDHungarianTracker
5) MPNTracker - with message passing network (graph neural nets) для целей уточнения предсказаний

На датасете market обучены несколько reid моделей с
- triplet loss / cosine loss
- hard negative mining

# Results 
### GIF
![detector result gif](/output/figs/MOT16-02_result.gif)

How to run
1. train reid net on market data
2. train gnn net on crops
3. use tracker

#  python -m eval_tracker --device='cuda:1' 
