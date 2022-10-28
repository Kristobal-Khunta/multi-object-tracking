import torch
from market import utils
from market import metrics


def evaluate(model, test_loader, metric_fn, ranks=[1, 5, 10, 20]):
    with torch.no_grad():
        model.eval()
        print("Extracting features from query set...")
        q_feat, q_pids, q_camids = utils.extract_features(model, test_loader["query"])
        print("Done, obtained {}-by-{} matrix".format(q_feat.size(0), q_feat.size(1)))

        print("Extracting features from gallery set ...")
        g_feat, g_pids, g_camids = utils.extract_features(model, test_loader["gallery"])
        print("Done, obtained {}-by-{} matrix".format(g_feat.size(0), g_feat.size(1)))

        distmat = metrics.compute_distance_matrix(q_feat, g_feat, metric_fn=metric_fn)
        distmat = distmat.numpy()

        print("Computing CMC and mAP ...")
        cmc, mAP = metrics.eval_market1501(
            distmat, q_pids, g_pids, q_camids, g_camids, max_rank=50
        )

        print("** Results **")
        print("mAP: {:.1%}".format(mAP))
        print("CMC curve")
        for r in ranks:
            print("Rank-{:<3}: {:.1%}".format(r, cmc[r - 1]))
        return cmc[0], mAP