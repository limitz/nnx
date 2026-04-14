import torch as _torch


def confusion(pred, target):
    pred = pred.long().flatten()
    target = target.long().flatten()
    tp = ((pred == 1) & (target == 1)).sum().float()
    tn = ((pred == 0) & (target == 0)).sum().float()
    fp = ((pred == 1) & (target == 0)).sum().float()
    fn = ((pred == 0) & (target == 1)).sum().float()
    return tp, tn, fp, fn


def accuracy(pred, target):
    tp, tn, fp, fn = confusion(pred, target)
    return (tp + tn) / (tp + tn + fp + fn)


def precision(pred, target):
    tp, tn, fp, fn = confusion(pred, target)
    denom = tp + fp
    if denom == 0:
        return _torch.tensor(0.0)
    return tp / denom


def recall(pred, target):
    tp, tn, fp, fn = confusion(pred, target)
    denom = tp + fn
    if denom == 0:
        return _torch.tensor(0.0)
    return tp / denom


def f1(pred, target):
    p = precision(pred, target)
    r = recall(pred, target)
    denom = p + r
    if denom == 0:
        return _torch.tensor(0.0)
    return 2 * p * r / denom


def matthews(pred, target):
    tp, tn, fp, fn = confusion(pred, target)
    numer = tp * tn - fp * fn
    denom = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)).sqrt()

    if denom == 0:
        return _torch.tensor(0.0)
    return numer / denom


def spearman(pred, target):
    """Spearman rank correlation via Pearson correlation of ranks."""
    pred = pred.float().flatten()
    target = target.float().flatten()

    def _rank(x):
        order = x.argsort()
        ranks = _torch.empty_like(x)
        ranks[order] = _torch.arange(1, len(x) + 1, dtype=x.dtype, device=x.device)
        # average ties
        unique_vals = x.unique()
        for v in unique_vals:
            mask = x == v
            if mask.sum() > 1:
                ranks[mask] = ranks[mask].mean()
        return ranks

    rpred = _rank(pred)
    rtarget = _rank(target)

    # Pearson on ranks
    rpred = rpred - rpred.mean()
    rtarget = rtarget - rtarget.mean()

    numer = (rpred * rtarget).sum()
    denom = rpred.pow(2).sum().sqrt() * rtarget.pow(2).sum().sqrt()

    if denom == 0:
        return _torch.tensor(0.0)
    return numer / denom
