import torch


class LpLoss(object):
    def __init__(
        self, mask, d=2, p=2, size_average=True, reduction=True, ignore_index=-32767
    ):
        super(LpLoss, self).__init__()
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.ignore_index = ignore_index
        self.mask = mask

    def masked_rel(self, x, y):
        num_examples = x.size()[0]
        mask_sliced = self.mask
        mask_expanded = mask_sliced.unsqueeze(0).unsqueeze(0)
        mask_re = mask_expanded.expand_as(x)
        mask = mask_re.reshape(num_examples, -1)
        diff = (x.view(num_examples, -1) - y.view(num_examples, -1)) * mask
        diff_norms = torch.norm(diff, self.p, 1)
        y_norms = torch.norm(y.view(num_examples, -1) * mask, self.p, 1)
        y_norms = y_norms + (y_norms == 0).float() * 1e-8
        rel_norms = diff_norms / y_norms
        if self.reduction:
            if self.size_average:
                return torch.mean(rel_norms)
            else:
                return torch.sum(rel_norms)

        return rel_norms

    def __call__(self, x, y):
        return self.masked_rel(x, y)
