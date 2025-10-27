""" Eval metrics and related

Hacked together by / Copyright 2020 Ross Wightman
"""
import torch

EVAL_VERIFICATION_RATES = [0.01, 0.02, 0.05, 0.1, 0.2]

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class CorrectnessOfPredictionsWithConfidencesMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.predictions_correct = []
        self.confidences = []

    def update(self, output, target):
        confidences, preds = output.topk(k=1)
        preds = preds.t()
        correct = preds.eq(target.reshape(1, -1).expand_as(preds)).flatten()

        self.predictions_correct.append(correct.detach().cpu())
        self.confidences.append(confidences.detach().cpu())

    def final_accuracy(self, vrs):
        correct = torch.cat(self.predictions_correct)
        confidences = torch.cat(self.confidences)

        correct_sorted = correct[confidences.flatten().argsort()]
        N = len(correct_sorted)

        def _fa(vr):
            n_verified = round(vr * N)
            return (n_verified + correct_sorted[n_verified:].sum()) / N

        return [_fa(vr) for vr in vrs]
    
    def average_final_accuracy(self, vrs):
        correct = torch.cat(self.predictions_correct)
        confidences = torch.cat(self.confidences)

        correct_sorted = correct[confidences.flatten().argsort()]
        N = len(correct_sorted)

        def _afa(vr):
            # see https://drive.google.com/file/d/1Uag8VtD3RwsoS8hs59X6T5u_iwuqspkS/view
            # for derivation of this formula
            n_verified = round(vr * N)
            afa_weights = torch.arange(1, N + 1) / n_verified
            return (
                (n_verified - 1) / 2
                + (afa_weights[:n_verified] * correct_sorted[:n_verified]).sum()
                + correct_sorted[n_verified:].sum()
            ) / N

        return [_afa(vr) for vr in vrs]


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = min(max(topk), output.size()[1])
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    return [correct[:min(k, maxk)].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
