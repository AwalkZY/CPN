import torch
import math
import torch.nn.functional as F


def analog_normal_dist(length, mean, std):
    """
    :param length: int, the total number of elements
    :param mean: int, in size (batch_size), the mean value of distributions
    :param std: float, in size (batch_size), the std value of distributions
    :return:
    """
    mean = mean.view(-1, 1).float()
    std = std.view(-1, 1).float()
    index = torch.arange(length).view(1, -1).float().to(mean.device)
    expo = - (index - mean) ** 2 / (2.0 * std ** 2)
    coef = 1.0 / math.sqrt(2 * math.pi) / std
    result = coef * torch.exp(expo)
    return result / torch.sum(result, dim=-1, keepdim=True)


def construct_target_cls(gt_start, gt_end, batch_size, max_len):
    """gt_start and gt_end in size (batch_size)"""
    target_cls = torch.zeros(batch_size, max_len + 2)  # in (batch_size, max_len)
    batch_idx = torch.arange(batch_size)
    target_cls[batch_idx, torch.round(gt_start).long()] = 1
    target_cls[batch_idx, torch.round(gt_end).long() + 1] = -1
    return target_cls.cumsum(dim=-1)[:, :max_len]


def calc_cls_loss(pred_cls, gt_start, gt_end):
    target_cls = construct_target_cls(gt_start, gt_end, pred_cls.size(0), pred_cls.size(1)).to(pred_cls.device)
    pos_loss = -(target_cls * (pred_cls + 1e-9).log()).mean()
    neg_loss = -((1 - target_cls) * (1 - pred_cls + 1e-9).log()).mean()
    return pos_loss + neg_loss


def calc_order_loss(order_pred, order_target):
    return F.cross_entropy(order_pred, order_target)


def calc_boundary_loss_group(all_start, all_end, gt_start, gt_end, bias, num_updates):
    batch_size, chunk_num, max_len = all_start.size()
    gt_std = (1.0 + (gt_end - gt_start + 1) / max_len)
    gt_start_dist = analog_normal_dist(max_len, gt_start, gt_std).to(all_start.device) \
        .unsqueeze(1).repeat(1, all_start.size(1), 1).view(-1, max_len)
    gt_end_dist = analog_normal_dist(max_len, gt_end, gt_std).to(all_start.device) \
        .unsqueeze(1).repeat(1, all_end.size(1), 1).view(-1, max_len)
    start_loss = F.kl_div(all_start.view(-1, max_len), gt_start_dist, reduction="batchmean")
    end_loss = F.kl_div(all_end.view(-1, max_len), gt_end_dist, reduction="batchmean")
    return start_loss + end_loss
    # """gt_start & gt_end : (batch_size), all_start & all_end : (batch_size, chunk_num, max_len)"""
    # all_start_stack, all_end_stack = all_start.view(-1, max_len), all_end.view(-1, max_len)
    # gt_start_stack = torch.round(gt_start.view(-1, 1).repeat(1, chunk_num).view(-1)).long().to(all_start_stack.device)
    # gt_end_stack = torch.round(gt_end.view(-1, 1).repeat(1, chunk_num).view(-1)).long().to(all_start_stack.device)
    # start_loss = F.cross_entropy(all_start_stack, gt_start_stack)
    # end_loss = F.cross_entropy(all_end_stack, gt_end_stack)
    # return start_loss + end_loss


def calc_diversity_loss(real_start, real_end, all_start, all_end):
    max_len = all_start.size(-1)
    group_real_start = real_start.unsqueeze(1).repeat(1, all_start.size(1), 1).view(-1, max_len)
    group_real_end = real_end.unsqueeze(1).repeat(1, all_end.size(1), 1).view(-1, max_len)
    start_div = torch.cosine_similarity(group_real_start, all_start.view(-1, max_len), dim=-1).mean()
    end_div = torch.cosine_similarity(group_real_end, all_end.view(-1, max_len), dim=-1).mean()
    return start_div + end_div


def calc_negative_loss(fake_start, fake_end):
    return fake_start.softmax(-1).max(dim=-1)[0].mean() + fake_end.softmax(-1).max(dim=-1)[0].mean()


def calc_variance_loss(all_start, all_end):
    # all_start / all_end in (bs, chunk_num, max_len)
    max_len = all_start.size(-1)
    accum_diff = 0
    for dist in range(4):
        accum_diff += ((all_start[:, :, dist:] - all_start[:, :, :max_len - dist]) ** 2).mean()
        accum_diff += ((all_end[:, :, dist:] - all_end[:, :, :max_len - dist]) ** 2).mean()
    return accum_diff


# def calc_variance_loss(features):
#     # features in (bs * chunk_num, max_len, hidden_dim)
#     max_len = features.size(1)


def calc_localization_loss(real_start, real_end, fake_start, fake_end, all_start, all_end, inside_prob, order_pred,
                           order_target, gt_start, gt_end, config, num_updates):
    # FIXME: Image-to-Real gt_end - 1
    boundary_loss = calc_boundary_loss_group((all_start + 1e-9).log(), (all_end + 1e-9).log(),
                                             gt_start, gt_end - 1, config["norm_bias"], num_updates)
    diversity_loss = calc_diversity_loss(real_start, real_end, all_start, all_end)
    variance_loss = calc_variance_loss(all_start, all_end)
    inside_loss = calc_cls_loss(inside_prob, gt_start, gt_end)
    total_loss = config["boundary"] * boundary_loss + config["inside"] * inside_loss
    # + variance_loss  # + negative_loss + 0.5 * diversity_loss  # + 0.1 * variance_loss + order_loss
    return total_loss, {
        "loss": total_loss.item(),
        "boundary": boundary_loss.item(),
        "inside": inside_loss.item(),
        "diversity": diversity_loss.item(),
        "variance": variance_loss.item()
        # "order_loss": order_loss.item()
    }


def calc_boundary_loss(real_start, real_end, gt_start, gt_end):
    """
    :param real_start: the predicted start position LOGIT, in (batch_size, max_len), It's NATURALLY NORMALIZED.
    :param real_end: the predicted end position LOGIT, in (batch_size, max_len), It's NATURALLY NORMALIZED.
    :param gt_start: the ground_truth start position, in (batch_size)
    :param gt_end: the ground_truth end position, in (batch_size)
    :return:
    """
    max_len = real_start.size(-1)
    gt_start_dist = analog_normal_dist(max_len, gt_start, 1.0).to(real_start.device)
    gt_end_dist = analog_normal_dist(max_len, gt_end, 1.0).to(real_start.device)
    start_loss = F.kl_div(real_start, gt_start_dist, reduction="batchmean")
    end_loss = F.kl_div(real_end, gt_end_dist, reduction="batchmean")
    return start_loss + end_loss


def calc_qa_loss(avg_pred, all_pred, target):
    batch_size, group_num, class_num = all_pred.size()
    all_target = target.unsqueeze(1).repeat(1, group_num).view(-1).to(all_pred.device)
    all_pred = all_pred.view(-1, class_num)
    all_ce_loss = F.cross_entropy(all_pred, all_target)
    avg_ce_loss = F.cross_entropy(avg_pred, target.to(avg_pred.device))
    total_loss = all_ce_loss + avg_ce_loss
    return total_loss, {
        "loss": total_loss.item(),
        "all_ce": all_ce_loss.item(),
        "avg_ce": avg_ce_loss.item()
    }
