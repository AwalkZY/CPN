import math

import torch
from utils.helper import no_peak_mask
from utils.text_processor import Vocabulary

def init_vars(model, source, source_mask, target_vocab, k, max_len):
    # assert source.size(0) == 1, "Invalid Number of sentences!"
    assert type(target_vocab) is Vocabulary, "Invalid type of target vocabulary!"
    batch_size = source.size(0)
    # Target sentence begins with Begin of Sentence
    init_token = target_vocab.stoi('<BOS>')
    target = torch.tensor([[init_token]] * batch_size).long()
    target_mask = no_peak_mask(1).repeat(batch_size, 1, 1)
    result, scores, encoding = model.forward(source, source_mask, target, target_mask, positional_encoding=True)
    # scores in shape (batch_size, sentence_length, vocab_size)
    # Here we do the slice operation on the "sentence_length" dimension
    k_scores, idx = scores[:, -1].data.topk(k)
    log_scores = torch.log(k_scores)
    # log_scores in shape (batch_size, k)
    target = torch.zeros(batch_size, k, max_len).long()
    target[:, :, 0] = init_token
    # Info: idx in size(batch_size, k)
    target[:, :, 1] = idx
    encodings = encoding.unsqueeze(1).repeat(1, k, 1, 1)
    # In shape (batch_size, k, max_len, emb_dim)
    return target, encodings, log_scores


def find_k_best(targets, scores, last_scores, i, k):
    # target in shape (batch_size, k, sentence_length)
    # scores in shape (batch_size, k, sentence_length, target_vocab)
    word_k_scores, word_idx = scores[:, :, -1].data.topk(k)
    # k_scores & idx are in shape (batch_size, k, k)
    # the second dimension: k candidate sentences, third dimension: k candidate words
    cum_log_scores = torch.log(word_k_scores) + last_scores.unsqueeze(-1)
    # use "log" operation to transform multiplication into logarithmic addition
    group_k_scores, group_idx = cum_log_scores.view(-1, k*k).topk(k)
    row = group_idx // k
    col = group_idx % k
    # Regenerate sentence sequence
    batch_idx = torch.arange(targets.size(0)).view(-1, 1)
    targets[batch_idx, :, :i] = targets[batch_idx, row, :i]
    targets[batch_idx, :, i] = word_idx[batch_idx, row, col]
    cum_log_scores = group_k_scores
    # cum_log_scores in shape (batch_size, k)
    return targets, cum_log_scores


def beam_search(model, source, source_mask, target_vocab, k, max_len):
    batch_size = source.size(0)
    alpha = 0.7
    targets, encodings, cum_log_scores = init_vars(model, source, source_mask, target_vocab, k, max_len)
    eos_token = target_vocab.stoi("<EOS>")
    idx = None
    for i in range(2, max_len):
        target_mask = no_peak_mask(i).repeat(batch_size, 1, 1)
        result, scores, encoding = model.forward(source, source_mask, targets[:, :i], target_mask, positional_encoding=True)
        targets, cum_log_scores = find_k_best(targets, scores, cum_log_scores, i, k)
        end_pos = (targets == eos_token).nonzero()
        # end_pos in shape (nonzero_num, 3)
        sentence_lengths = torch.zeros(batch_size, targets.size(1), dtype=torch.long).to(source.device)
        for pos in end_pos:
            batch_i, beam_i = pos[0], pos[1]
            if sentence_lengths[batch_i, beam_i] == 0:
                sentence_lengths[batch_i, beam_i] = pos[2]
        finished_num = torch.sum(sentence_lengths > 0, dim=1)
        if (finished_num == k).all():
            ans = cum_log_scores / (sentence_lengths.type_as(cum_log_scores) ** alpha)
            _, idx = torch.max(ans, dim=1)
            break
    if idx is None:
        is_finished = torch.gt(sentence_lengths, 0)
        ans = cum_log_scores / (sentence_lengths.type_as(cum_log_scores) ** alpha)
        ans.masked_fill_(is_finished, 0)
        _, idx = torch.max(ans, dim=1)
        # idx in shape (batch_size)
    batch_idx = torch.arange(batch_size)
    target_token = targets[batch_idx, idx]
    target_scores = cum_log_scores[batch_idx, idx]
    target_mask = ((targets[batch_idx, idx] == eos_token).cumsum(dim=-1) == 0)
    target_token.masked_fill_(target_mask == 0, target_vocab.stoi("<PAD>"))
    target_scores.masked_fill_(target_mask == 0, 0)
    target_length = target_mask.sum(dim=-1)
    # sentence = ' '.join([
    #     target_vocab.itos[token] for token in targets[idx][1:length]
    # ])
    return target_token, target_scores, target_length



