import torch
import torch.nn as nn
import torch.nn.functional as F


class Loss():
    def __init__(self):
        pass

    def get_loss_fn(self):
        
        class_loss_fn = self.class_loss_fn

        def loss_fn(outputs, labels, num_items_in_batch=None):
            
            labels = nn.functional.pad(labels, (0, 1), value=-100)
            shift_labels = labels[..., 1:].contiguous()
            loss_mask = shift_labels != -100
            shift_labels[~loss_mask] = 0

            per_token_loss = class_loss_fn(outputs.logits, shift_labels)

            if num_items_in_batch is None:
                num_items_in_batch = loss_mask.sum()
            
            loss = (per_token_loss * loss_mask).sum() / num_items_in_batch
            return loss
        
        return loss_fn
    
    def class_loss_fn(self, logits, labels):
        raise NotImplementedError("Subclasses must implement class_loss!")
    

class FocalLoss(Loss):
    def __init__(self, gamma=3.0):
        super().__init__()
        self.gamma = gamma

    def class_loss_fn(self, logits, targets):
        
        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        p_t = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        log_p_t = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        focal_term = (1 - p_t) ** self.gamma
        loss = -focal_term * log_p_t

        return loss

    

class TOFULoss(Loss):
    def __init__(self, gamma=3.0, beta=0.8):
        super().__init__()
        self.gamma = gamma
        self.beta = beta

    def class_loss_fn(self, logits, targets):
        
        log_probs_beta = F.log_softmax(logits / self.beta, dim=-1)
        probs = F.log_softmax(logits, dim=-1).exp()

        p = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
        log_p_b = log_probs_beta.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        term = ((1-p)**self.gamma - self.gamma*p*(1-p)**(self.gamma-1)*torch.log(p)).detach()
        loss = -term * log_p_b * self.beta

        return loss



# PR loss are from https://www.arxiv.org/abs/2508.09654 ######

class PRLoss(Loss):
    def __init__(self, gamma=1e-5, lam=1e-1):
        super().__init__()
        self.gamma = gamma  # default ranges: gamma in [1e-7, 1e-5]
        self.lam = lam      # lambda in [1e-1, 1]

    def class_loss_fn(self, logits, targets):

        L = logits.shape[1]

        log_probs = F.log_softmax(logits, dim=-1)
        probs = log_probs.exp()

        # Get probabilities for target tokens
        Q_vals = probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)  # (B, L)
        log_Q_vals = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)

        # Detached version Q bar
        Q_bar = Q_vals.detach()

        # Compute delta
        delta = self.gamma * (self.lam**(1/L)) / (1.0 - (1 - self.gamma) * (self.lam**(1/L)))

        # Compute weight function w(l, lambda, gamma)
        weights = []
        for l in range(L):
            power_term = self.lam ** (l / max(L - 1, 1))  # Avoid division by zero
            mask = (Q_vals[:, l] <= delta).float()
            frac = Q_bar[:, l] / (self.gamma + (1 - self.gamma) * Q_bar[:, l])
            w = power_term * mask * frac
            weights.append(w)
        weights = torch.stack(weights, dim=1)  # (B, L)

        # Final loss
        loss = -(weights * log_Q_vals)
        
        return loss

# GEM loss from https://arxiv.org/abs/2408.16673

class GEMLoss(Loss):
    def __init__(self, beta=0.7, h="linear"):
        super().__init__()
        self.beta = beta
        self.h = h

    def class_loss_fn(self, logits, labels):

        with torch.no_grad():
            logits_on_labels = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            logits_diff = logits - logits_on_labels.unsqueeze(-1)

            if self.h == "linear":
                weights = torch.ones_like(logits_diff)
            elif self.h == "log_sigmoid":
                weights = torch.sigmoid(0.01 * logits_diff)
            else:
                raise ValueError(f"Invalid h: {self.h}")

        gene_log_probs = F.log_softmax(logits, dim=-1)
        q_probs = torch.exp(F.log_softmax(logits / self.beta, dim=-1)).detach()

        real_log_probs = torch.gather(gene_log_probs, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)

        loss = -(q_probs * weights * (real_log_probs.unsqueeze(-1) - gene_log_probs)).sum(-1)

        return loss

