import sys
import torch
from progressbar import *
from utils.helper import move_to_cuda
import pandas as pd


def question_answer(model, data_loader, name, answer_path, **kwargs):
    answer_set = pd.read_csv(answer_path, header=None)[0]
    correct = 0
    pbar = ProgressBar(widgets=['Progress: ', Percentage(), ' ', Bar('#'), ' ', Timer(),
                                ' ', ETA()]).start()
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, 1):
            pbar.update(int((batch_idx / len(data_loader)) * 100))
            net_input = move_to_cuda(batch['net_input'])
            output = model(**net_input)
            result = output["avg_result"].squeeze().argmax(dim=-1).long()
            for pred, target in zip(result, batch["raw"]):
                if answer_set[pred.item()] == target[0]:
                    correct += 1
        pbar.finish()
        return 1.0 * correct / len(data_loader) / data_loader.batch_size
