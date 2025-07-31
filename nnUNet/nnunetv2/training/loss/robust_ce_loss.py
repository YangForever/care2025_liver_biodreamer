import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            print(target.shape)
            print(input.shape)
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())
    
class RobustCrossEntropyLoss_Weighted(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def __init__(self, *args, **kwargs):
        super(RobustCrossEntropyLoss_Weighted, self).__init__(*args, **kwargs)
        #self.weights = weights # list of weights for each class

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
    
# testing added by Yang
if __name__ == '__main__':
    normal_loss = RobustCrossEntropyLoss()
    ce_kwargs = {}
    ce_kwargs['ignore_index'] = 0
    ce_kwargs['weight'] = torch.Tensor([0.1, 0.9])
    weighted_loss = RobustCrossEntropyLoss_Weighted(**ce_kwargs)
    i = torch.rand(1,2,3,3)
    p = torch.Tensor([[[1, 0, 1], [1, 0, 1], [1, 0, 1]]])
    print(i)
    print(p)
    print(i.shape)
    print(p.shape)
    print(normal_loss(i, p))
    print(weighted_loss(i, p))

