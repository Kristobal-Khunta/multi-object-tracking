import torch
from market import utils
from market import metrics


def evaluate(model, test_loader, metric_fn, ranks=[1, 5, 10, 20]):
    with torch.no_grad():
        model.eval()
        print("Extracting features from query set...")
        q_feat, q_pids, q_camids = utils.extract_features(model, test_loader["query"])
        print(f"Done, obtained {q_feat.size(0)}-by-{q_feat.size(1)} matrix")

        print("Extracting features from gallery set ...")
        g_feat, g_pids, g_camids = utils.extract_features(model, test_loader["gallery"])
        print(f"Done, obtained {g_feat.size(0)}-by-{g_feat.size(1)} matrix")

        distmat = metrics.compute_distance_matrix(q_feat, g_feat, metric_fn=metric_fn)
        distmat = distmat.numpy()

        print("Computing CMC and mAP ...")
        cmc, mAP = metrics.eval_market1501(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50
        )

        print("** Results **")
        print(f"mAP: {mAP:.1%}")
        print("CMC curve")
        for r in ranks:
            print(f"Rank-{r:<3}: {cmc[r - 1]:.1%}")
        return cmc[0], mAP
