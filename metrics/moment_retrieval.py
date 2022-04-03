import random
import sys

import numpy as np
import torch
import torch.nn.functional as F

from utils.calculator import calculate_iou1d
from utils.container import metricsContainer
from utils.helper import move_to_cuda
from utils.slimer import expand_and_repeat


def calc_boundary_prob(real_start, real_end):
    return real_start.unsqueeze(-1) * real_end.unsqueeze(1)


def calc_in_prob(pred_in):
    expanded_pred_in = torch.zeros(pred_in.size(0), pred_in.size(1) + 1).to(pred_in.device)
    expanded_pred_in[:, :pred_in.size(1)] = pred_in
    expanded_pred_in = expanded_pred_in.cumsum(-1)
    return expanded_pred_in.unsqueeze(1) - expanded_pred_in.unsqueeze(-1)


def get_best_prop(boundary_prob):
    batch_size, max_len, _ = boundary_prob.size()
    idx = torch.arange(max_len).to(boundary_prob.device)
    start_idx, end_idx = idx.view(1, -1, 1), idx.view(1, 1, -1)
    # in (batch_size, max_len, max_len)
    joint_prob = boundary_prob.masked_fill(end_idx - start_idx < 0, 0.0).view(batch_size, -1)
    best_prop = joint_prob.max(-1)[1]
    best_prop = torch.stack((best_prop // max_len, best_prop % max_len + 1), dim=-1)
    # FIXME: Image-to-Real + 1
    return best_prop


def calc_pos_iou(max_len, first_pos, second_pos):
    first_start, first_end = first_pos // max_len, first_pos % max_len + 1
    second_start, second_end = second_pos // max_len, second_pos % max_len + 1
    iou = calculate_iou1d(first_start, first_end, second_start, second_end)
    return iou


def get_best_n_props(boundary_prob, n, thresh):
    batch_size, max_len, _ = boundary_prob.size()
    idx = torch.arange(max_len).to(boundary_prob.device)
    start_idx, end_idx = idx.view(1, -1, 1), idx.view(1, 1, -1)
    # in (batch_size, max_len, max_len)
    joint_probs = boundary_prob.masked_fill(end_idx - start_idx < 0, 0.0).view(batch_size, -1)
    # in (batch_size, max_len * max_len)
    sorted_prob, sorted_idx = joint_probs.sort(dim=-1, descending=True)
    # both in (batch_size, max_len * max_len)
    result = np.zeros((batch_size, n))
    for batch_idx in range(batch_size):
        result[batch_idx, 0] = sorted_idx[batch_idx][0]
        valid_mask = sorted_prob[batch_idx] > 0.0
        valid_prob, valid_idx = sorted_prob[batch_idx][valid_mask].cpu().numpy(), sorted_idx[batch_idx][
            valid_mask].cpu().numpy()
        # (valid_num)
        for chosen_idx in range(1, n):
            last_result = result[batch_idx, chosen_idx - 1].repeat(valid_idx.shape[0])[np.newaxis, :]
            other_candidates = valid_idx[np.newaxis, :]
            iou = calc_pos_iou(max_len, last_result, other_candidates).squeeze()
            valid_mask = iou <= thresh
            if np.sum(valid_mask) >= n - chosen_idx:
                valid_prob, valid_idx = valid_prob[valid_mask], valid_idx[valid_mask]
            result[batch_idx, chosen_idx] = valid_idx[0]
            assert (valid_prob[0] >= valid_prob).all(), "Invalid Choice of results!"
    prop_result = np.zeros((batch_size, n, 2))
    prop_result[:, :, 0], prop_result[:, :, 1] = result // max_len, result % max_len + 1
    prop_result = prop_result.transpose((1, 0, 2))
    return prop_result


def get_ensemble_result(net_output):
    boundary_prob = calc_boundary_prob(net_output["real_start"], net_output["real_end"])
    best_prop = get_best_prop(boundary_prob)
    if net_output["fake_start"] is not None:
        fake_boundary_prob = calc_boundary_prob(net_output["fake_start"], net_output["fake_end"])
        fake_prop = get_best_prop(fake_boundary_prob)
    else:
        fake_boundary_prob = None
        fake_prop = None
    return best_prop, boundary_prob, fake_prop, fake_boundary_prob


def get_individual_result(net_output):
    all_start, all_end = net_output["all_start"], net_output["all_end"]
    best_prop_list = []
    boundary_prob_list = []
    for i in range(all_start.size(1)):
        boundary_prob = calc_boundary_prob(all_start[:, i], all_end[:, i])
        best_prop = get_best_prop(boundary_prob)
        best_prop_list.append(best_prop)
        boundary_prob_list.append(boundary_prob)
    return best_prop_list, boundary_prob_list


def moment_retrieval(model, data_loader, top_n, thresh, by_frame, display_interval):
    """
    :param display_interval: the interval of displaying the prediction and ground-truth
    :param data_loader: the data loader providing data
    :param model: the model to be evaluated
    :param top_n: rank parameter
    :param thresh: the parameter controlling how to choose predictions, only valid when top_n > 1
    :param by_frame: how to calculate the iou value
    :return: dict[float]
    """
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, 1):
            net_input = move_to_cuda(batch["net_input"])
            net_output = model(**net_input)
            best_prop, boundary_prob, fake_prop, fake_boundary_prob = get_ensemble_result(net_output)
            best_n_prop = get_best_n_props(boundary_prob, top_n, thresh)
            best_prop_list, boundary_prob_list = get_individual_result(net_output)
            batch_size = best_prop.size(0)
            target = np.stack((batch['target']['start'], batch['target']['end']), axis=-1)
            if batch_idx % display_interval == 0:
                print("-" * 20)
                chosen_idx = np.random.randint(batch_size)
                best_pred = best_prop.cpu().tolist()[chosen_idx]
                best_prob = boundary_prob[chosen_idx, best_pred[0], best_pred[1] - 1].item()
                gt_prob = boundary_prob[
                    chosen_idx, int(target[chosen_idx][0]), int(target[chosen_idx][1]) - 1].item()
                print("Ens Pred Bound: {}, True Bound: {}".format(best_pred, target[chosen_idx]))
                print("Ens Pred Prob: {:.4f}, True Prob: {:.4f}".format(best_prob, gt_prob))
                ind_prop_msg, ind_prob_msg = "Ind Pred Bound: ", "Ind Pred Prob: "
                for i in range(len(best_prop_list)):
                    best_pred_item = best_prop_list[i].cpu().tolist()[chosen_idx]
                    ind_prop_msg += str(best_pred_item) + " | "
                    ind_prob_msg += str(boundary_prob_list[i][chosen_idx, best_pred_item[0],
                                                              best_pred_item[1] - 1].item())[:5] + " | "
                print(ind_prop_msg)
                print(ind_prob_msg)
                print("-" * 20)
                sys.stdout.flush()
            metricsContainer.update("top_1", top_1_metric(best_prop.cpu().numpy(), target, by_frame))
            # metricsContainer.update("top_{}".format(top_n), top_n_metric(best_n_prop, target, by_frame))
        return {
            "top_1": metricsContainer.calculate_average("top_1"),
            # "top_{}".format(top_n): metricsContainer.calculate_average("top_{}".format(top_n)),
        }


def qualitative_moment_retrieval(model, data_loader, top_n, min_thresh, max_thresh, min_length, max_length, min_start, max_end,
                                 max_number, by_frame, display_interval, **kwargs):
    collected_result = []
    counter = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader, 1):
            net_input = move_to_cuda(batch["net_input"])
            net_output = model(**net_input)
            best_prop, boundary_prob, fake_prop, fake_boundary_prob = get_ensemble_result(net_output)
            targets = np.stack((batch['target']['start'], batch['target']['end']), axis=-1)
            raw = batch['raw']
            predictions = best_prop.cpu().numpy()
            for i, (prediction, target, raw_info) in enumerate(zip(predictions, targets, raw)):
                iou = calculate_single_IoU((prediction[0], prediction[1]), (target[0], target[1]))
                counter += 1
                # if raw_info[0] == "v_IjULOynkK5I" and raw_info[-1] == " She is instructing a class.":
                if min_thresh <= iou <= max_thresh and min_length <= prediction[1] - prediction[0] <= max_length and prediction[0] >= min_start and prediction[1] <= max_end:
                    if random.random() > 0.8:
                        continue
                    print("{} is collected! Progress: {}%".format(raw_info[0],
                                                                  100.0 * counter / len(data_loader) / data_loader.batch_size))
                    sys.stdout.flush()
                    collected_result.append({
                        "pred_start": net_output["real_start"][i].cpu().numpy().tolist(),
                        "pred_end": net_output["real_end"][i].cpu().numpy().tolist(),
                        "true_start": target[0].item(),
                        "true_end": target[1].item(),
                        "inside": net_output["inside_prob"][i].cpu().numpy().tolist(),
                        "raw": raw_info
                    })
            if len(collected_result) > max_number:
                break
    return collected_result


def top_1_metric(pred, label, by_frame):
    result = {}
    bsz = pred.shape[0]
    iou = calculate_IoU((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]), by_frame)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def top_n_metric(pred, label, by_frame):
    result = {}
    bsz = pred[0].shape[0]
    top_iou = []
    for pred in pred:
        iou = calculate_IoU((pred[:, 0], pred[:, 1]), (label[:, 0], label[:, 1]), by_frame)
        top_iou.append(iou)
    iou = np.max(np.stack(top_iou, 1), 1)
    result['mIoU'] = np.mean(iou)
    for i in range(1, 10, 2):
        result['IoU@0.{}'.format(i)] = 1.0 * np.sum(iou >= i / 10) / bsz
    return result


def calculate_IoU(i0, i1, by_frame):
    union = (np.min(np.stack([i0[0], i1[0]], 0), 0), np.max(np.stack([i0[1], i1[1]], 0), 0))
    inter = (np.max(np.stack([i0[0], i1[0]], 0), 0), np.min(np.stack([i0[1], i1[1]], 0), 0))
    if by_frame:
        iou = 1.0 * (inter[1] - inter[0] + 1) / (union[1] - union[0] + 1)
    else:
        iou = 1.0 * (inter[1] - inter[0] + 1e-9) / (union[1] - union[0] + 1e-9)
    iou[union[1] - union[0] < -1e-5] = 0
    iou[iou < 0] = 0.0
    return iou


def calculate_single_IoU(i0, i1):
    union = (min(i0[0], i1[0]), max(i0[1], i1[1]))
    inter = (max(i0[0], i1[0]), min(i0[1], i1[1]))
    iou = 1.0 * (inter[1] - inter[0] + 1e-9) / (union[1] - union[0] + 1e-9)
    if union[1] - union[0] < -1e-5 or iou < 0:
        iou = 0.0
    return iou
