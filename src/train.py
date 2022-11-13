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
    parser.add_argument("--mode", type=str, default='classification', )
    parser.add_argument("--metric_fn", type=str, default= 'cosine', help="metric_fn")

    parser.add_argument("--max_epoch", type=int, default=30)
    parser.add_argument("--epoch_eval_freq", type=int, default=5)
    parser.add_argument("--print_freq", type=int, default=50)
    parser.add_argument("--help", "-h", action="help")
    return parser


def _select_model_head_from_args(args):
    if args.head_type == 'classification':
        model_head_loss = 'softmax'
    elif args.head_type == 'triplet':
        model_head_loss = 'triplet'
    else:
        raise ValueError('mode must be one of [classification, triplet]')
    return model_head

def _select_criterion_from_args(args):
    if args.mode == 'classification':
        criterion = torch.nn.CrossEntropyLoss()
    elif args.mode == 'triplet':
        criterion = CombinedLoss(0.3, 1.0, 1.0)
    else:
        raise ValueError('mode must be one of [classification, triplet]')
    return criterion
    
def _select_metric_fn_from_args(args):
    if args.metric_fn == 'cosine':
        metric_fn = cosine_distance
    elif args.metric_fn == 'euclidian':
        metric_fn = euclidean_squared_distance
    else:
        raise ValueError('metric_fn must be one of [cosine, euclidian]')

def main():
    set_all_seeds(12347)
    parser = setup_parser()
    args = parser.parse_args()
    metric_fn = _select_metric_fn_from_args(args)
    criterion = _select_criterion_from_args(args)
    model_head_loss_type = _select_model_head_from_args(args)
    reid_root_dir = ".."
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
        "resnet34", datamanager.num_train_pids, loss=model_head_loss_type, pretrained=True
    )
    model = model.cuda()

    #### Define optimizer and scheduler #####
    trainable_params = model.parameters()
    optimizer = torch.optim.Adam(
        trainable_params, lr=0.0003, weight_decay=5e-4, amsgrad=True
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)
    #metric_fn = cosine_distance  # cosine_distance or euclidean_squared_distance
    

    num_batches = len(train_loader)
    criterion = torch.nn.CrossEntropyLoss()

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
            if (batch_idx + 1) % PRINT_FREQ == 0:
                utils.print_statistics(
                    batch_idx, num_batches, epoch, args.max_epoch, batch_time, losses
                )
            end = time.time()

        if (epoch + 1) % EPOCH_EVAL_FREQ == 0 or epoch == args.max_epoch - 1:
            rank1, mAP = evaluate(model, test_loader, metric_fn=metric_fn)
            print(
                "Epoch {0}/{1}: Rank1: {rank}, mAP: {map}".format(
                    epoch + 1, args.max_epoch, rank=rank1, map=mAP
                )
            )


if __name__ == "__main__":
    main()



