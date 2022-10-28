import torch


class HardBatchMiningTripletLoss(torch.nn.Module):
    """Triplet loss with hard positive/negative mining of samples in a batch.

    Reference:
        Hermans et al. In Defense of the Triplet Loss for Person Re-Identification. arXiv:1703.07737.
    Args:
        margin (float, optional): margin for triplet. Default is 0.3.
    """

    def __init__(self, margin=0.3):
        super(HardBatchMiningTripletLoss, self).__init__()
        self.margin = margin
        self.ranking_loss = torch.nn.MarginRankingLoss(margin=margin)

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): feature matrix with shape (batch_size, feat_dim).
            targets (torch.LongTensor): ground truth labels with shape (batch_size).
        """
        n = inputs.size(0)

        # Compute the pairwise euclidean distance between all n feature vectors.

        # distance_matrix = euclidean_squared_distance(inputs, inputs)
        # distance_matrix = distance_matrix.clamp(min=1e-12).sqrt()
        distance_matrix = torch.cdist(inputs, inputs, p=2.0)  # clear euclidian dist

        # For each sample (image), find the hardest positive and hardest negative sample.
        # The targets are a vector that encode the class label for each of the n samples.
        # Pairs of samples with the SAME class can form a positive sample.
        # Pairs of samples with a DIFFERENT class can form a negative sample.
        #
        # loop over all samples, and for each one
        # find the hardest positive sample and the hardest negative sample.
        # The distances are then added to the following lists.
        # Positive pairs should be as close as possible, while
        # negative pairs should be quite far apart.
        mask = targets.expand(n, n).eq(targets.expand(n, n).t())

        distance_positive_pairs, distance_negative_pairs = [], []
        for i in range(n):
            row_dist = distance_matrix[i]
            row_mask = mask[i]
            hard_pos_dist = row_dist[row_mask].max().unsqueeze(0)
            hard_neg_dist = row_dist[row_mask == 0].min().unsqueeze(0)
            distance_positive_pairs.append(hard_pos_dist)
            distance_negative_pairs.append(hard_neg_dist)
        distance_positive_pairs = torch.cat(distance_positive_pairs)
        distance_negative_pairs = torch.cat(distance_negative_pairs)

        # The ranking loss will compute the triplet loss with the margin.
        # loss = max(0, -1*(neg_dist - pos_dist) + margin)
        y = torch.ones_like(distance_negative_pairs)
        return self.ranking_loss(distance_negative_pairs, distance_positive_pairs, y)
