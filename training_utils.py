import numpy as np
import torch
import torch.nn as nn


def lr_schedule(n_steps, lr_min, lr_max):
    """Generates a concave learning rate schedule over
    `n_steps`, starting and ending at `lr_min` and
    peaking at `lr_max` mid-way.
    """
    schedule = np.arange(n_steps)/(n_steps - 1)
    schedule = np.sin(schedule*np.pi)
    schedule = lr_min + (lr_max - lr_min)*schedule
    return schedule


def set_lr(optimizer, lr):
    """Sets learning rate for an optimizer"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def loss_fxn(yh, y, penalty_scale=1, device=None):
    """Binary cross entropy loss across each of the targets.

    Applies a penalty for average probability predictions that differ from observed frequency. 
    This penalty improves calibration of predicted probabilities, and aligns the model's 
    average prediction with its baseline prediction.

    Args:
        yh (dict): Classifier outputs from `MLBExplainerModel` prediction
        y (dict): Targets
        device (str, optional): Device model is on

    Returns:
        loss (torch.Tensor)
    """
    losses = []
    for k, targets in y.items():
        
        predictions = yh[k]                                # Probability predictions for target (n_obs, n_thresholds)
        targets = torch.tensor(targets, device=device)     # Observed outcomes for target (n_obs, n_thresholds)

        obs_freqs = torch.mean(targets, dim=0) + 1e-6      # Observed frequencies across each threshold of a target
        avg_probs = torch.mean(predictions, dim=0) + 1e-6  # Model's average probability prediction for each threshold of a target

        penalty = torch.mean(torch.abs(torch.log(avg_probs/obs_freqs)))
        loss = nn.BCELoss()(predictions, targets) + penalty*penalty_scale
        losses.append(loss)

    return sum(losses)/len(losses)


def baseline_penalty(yh_baseline, y, penalty_scale=1, device=None):
    """Penalty for probability predictions of baseline inputs that differ from observed frequency. 
    This penalty encourages the model's baseline prediction to match the observed avg frequency of the data set.

    Args:
        yh_baseline (dict): Classifier outputs from `MLBExplainerModel` prediction for baseline input
        y (dict): Targets
        penalty_scale (float): Weight to apply to penalty
        device (str, optional): Device model is on
    """
    pentalties = []
    for k, targets in y.items():
        baseline_probs = yh_baseline[k] + 1e-6
        targets = torch.tensor(targets, device=device)

        obs_freqs = torch.mean(targets, dim=0) + 1e-6      # Observed frequencies across each threshold of a target

        pentalty = torch.mean(torch.abs(torch.log(baseline_probs/obs_freqs)))
        pentalties.append(pentalty)
        
    return sum(pentalties)/len(pentalties)*penalty_scale