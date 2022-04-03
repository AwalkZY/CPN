import numpy as np

from lib import pybleu
from utils.calculator import calculate_iou1d


class LocalizationMetrics:
    @staticmethod
    def top_1_metric(pred_first, pred_last, target_first, target_last):
        result = {}
        batch_size = len(pred_first)
        iou = calculate_iou1d(pred_first, pred_last, target_first, target_last)
        result['mIoU'] = np.mean(iou)
        for i in np.arange(0.1, 1, 0.2):
            result['IoU@{:.1f}'.format(i)] = 1.0 * np.sum(iou >= i) / batch_size
        return result

    @staticmethod
    def top_n_metric(preds_first, preds_last, target_first, target_last):
        result = {}
        batch_size = len(preds_first[0])
        top_iou = []
        for idx in range(len(preds_first)):
            pred_first = preds_first[idx]
            pred_last = preds_last[idx]
            iou = calculate_iou1d(pred_first, pred_last, target_first, target_last)
            top_iou.append(iou)
        iou = np.max(np.stack(top_iou, 1), 1)
        result['mIoU'] = np.mean(iou)
        for i in np.arange(0.1, 1, 0.2):
            result['IoU@{:.1f}'.format(i)] = 1.0 * np.sum(iou >= i) / batch_size
        return result


class CaptionMetrics:

    @staticmethod
    def meteor(*inputs):
        pass

    @staticmethod
    def cider(*inputs):
        pass

    @staticmethod
    def wmd(*inputs):
        pass

    @staticmethod
    def bleu(ref, out):
        scorer = pybleu.PyBleuScorer()
        return scorer.score(ref, out)

    @staticmethod
    def perplexity(*inputs):
        pass

    @staticmethod
    def spice(*inputs):
        pass

    @staticmethod
    def rouge(*inputs):
        pass
