import torch
import torch.nn.functional as F


def tofu_loss(gamma=3.0, beta=0.8):
    def loss_fn(outputs, labels, num_items_in_batch=None):
        labels = F.pad(labels, (0, 1), value=-100)
        shift_labels = labels[..., 1:].contiguous()
        loss_mask = shift_labels != -100
        shift_labels[~loss_mask] = 0

        log_probs_beta = F.log_softmax(outputs.logits / beta, dim=-1)
        probs = F.log_softmax(outputs.logits, dim=-1).exp()

        p = probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        log_p_b = log_probs_beta.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)

        term = ((1-p)**gamma - gamma*p*(1-p)**(gamma-1)*torch.log(p)).detach()
        per_token_loss = -term * log_p_b * beta

        if num_items_in_batch is None:
            num_items_in_batch = loss_mask.sum()
        
        loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
        return loss
    return loss_fn

    