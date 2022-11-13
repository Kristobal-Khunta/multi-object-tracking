import time

import motmetrics as mm
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment as linear_assignment

from market import  utils

# Load helper code
from market.datamanager import ImageDataManager
from market.eval import evaluate
from market.models import build_model
import argparse
mm.lap.default_solver = "lap"
from mot.utils import euclidean_squared_distance, cosine_distance



def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    return None


def setup_parser():
    """Set up Python's ArgumentParser with data, model, trainer, and other arguments."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--model_head_type", type=str, default='softmax',choices=['softmax', 'triplet'] )

    parser.add_argument("--metric_fn", type=str, default= 'cosine', help="metric_fn")
    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--epoch_eval_freq", type=int, default=5)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--help", "-h", action="help")
    return parser


name2metricfn = {
    'cosine':cosine_distance,
    'euclidian':euclidean_squared_distance,
}

def main():
    set_all_seeds(12347)
    parser = setup_parser()
    args = parser.parse_args()
    
    reid_root_dir = ".."

    ### DATA ###
    datamanager = ImageDataManager(
        root=reid_root_dir,
        height=256,
        width=128,
        batch_size_train=32,
        workers=2,
        transforms=["random_flip", "random_crop"],
    )
    train_loader = datamanager.train_loader
    test_loader = datamanager.test_loader

    #### Define model ####
    model = build_model(
        "resnet34", datamanager.num_train_pids, loss=args.model_head_type, pretrained=True
    )
    model = model.cuda()

    #### Define optimizer and scheduler #####
    trainable_params = model.parameters()
    optimizer = torch.optim.Adam(
        trainable_params, lr=0.0003, weight_decay=5e-4, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

    #### criterion #####
    if args.model_head_type == 'softmax':
        criterion = torch.nn.CrossEntropyLoss()
    if args.model_head_type == 'triplet':
        criterion = CombinedLoss(0.3, 1.0, 1.0)

    ##### metric_fn #####
    metric_fn = name2metricfn[args.metric_fn]

    #### Train #######
    num_batches = len(train_loader)
    for epoch in range(args.max_epoch):
        losses = utils.MetricMeter()
        batch_time = utils.AverageMeter()
        end = time.time()
        model.train()
        
        for batch_idx, data in enumerate(train_loader):
            # Predict output.
            imgs, pids = data["img"].cuda(), data["pid"].cuda()
            output = model(imgs)
            # Compute loss.
            loss = criterion(output, pids)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            

            batch_time.update(time.time() - end)
            losses.update({"Loss": loss})
            if (batch_idx + 1) % args.print_freq == 0:
                utils.print_statistics(
                    batch_idx, num_batches, epoch, args.max_epoch, batch_time, losses
                )
            end = time.time()
        
        if (epoch + 1) % args.epoch_eval_freq == 0 or epoch == args.max_epoch - 1:
            rank1, mAP = evaluate(model, test_loader, metric_fn=metric_fn)
            print(
                "Epoch {0}/{1}: Rank1: {rank}, mAP: {map}".format(
                    epoch + 1, args.max_epoch, rank=rank1, map=mAP
                )
            )
        scheduler.step()
        
        


if __name__ == "__main__":
    main()



