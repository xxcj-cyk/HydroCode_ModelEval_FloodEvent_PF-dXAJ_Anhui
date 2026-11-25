import torch
from torch import nn


# ---------------------------------------------------------------------------
# Mask
# ---------------------------------------------------------------------------


def valid_train_mask(output, target, min_count=1):
    mask = ~torch.isnan(output) & ~torch.isnan(target)
    if mask.sum() < min_count:
        return None
    return mask


def zero_train_loss(output):
    return output.new_zeros((), requires_grad=True)


# ---------------------------------------------------------------------------
# Training loss
# ---------------------------------------------------------------------------


def rmse_from_diff(diff):
    return torch.sqrt(torch.mean(diff * diff))


class RMSELoss(nn.Module):
    def forward(self, output, target):
        mask = valid_train_mask(output, target, min_count=1)
        if mask is None:
            return zero_train_loss(output)
        return rmse_from_diff(output[mask] - target[mask])


class PFPeak80Loss(nn.Module):
    """Window-batch PF-Loss (non-event path).

    Per sample then batch-mean::

        L = overall_w * RMSE + win_w * RMSE_{Qobs >= γ·Qp}

    Event-stitched training (``event_loss_agg``) uses ``train.event_point_loss``
    instead, which also supports ``peak_excess`` / ``peak_underonly``.
    """

    def __init__(
        self,
        overall_weight=1.0,
        win_weight=0.5,
        peak_ratio=0.8,
        eps=1e-8,
    ):
        super().__init__()
        self.overall_weight = float(overall_weight)
        self.win_weight = float(win_weight)
        self.peak_ratio = float(peak_ratio)
        self.eps = float(eps)

    def forward(self, output, target):
        sim = output.squeeze(-1) if output.dim() == 3 else output
        obs = target.squeeze(-1) if target.dim() == 3 else target
        if sim.dim() == 1:
            sim = sim.unsqueeze(0)
            obs = obs.unsqueeze(0)

        if sim.dim() != 2:
            mask = valid_train_mask(output, target, min_count=1)
            if mask is None:
                return zero_train_loss(output)
            sim1 = output[mask]
            obs1 = target[mask]
            overall = torch.sqrt(torch.mean((sim1 - obs1) ** 2) + self.eps)
            peak = torch.max(obs1)
            if not torch.isfinite(peak):
                return overall
            high = obs1 >= self.peak_ratio * peak
            win = (
                torch.sqrt(torch.mean((sim1[high] - obs1[high]) ** 2) + self.eps)
                if high.any()
                else overall
            )
            return self.overall_weight * overall + self.win_weight * win

        finite = torch.isfinite(sim) & torch.isfinite(obs)
        if finite.sum() < 1:
            return zero_train_loss(output)

        obs_f = torch.where(finite, obs, torch.zeros_like(obs))
        sim_f = torch.where(finite, sim, torch.zeros_like(sim))
        sq = (sim_f - obs_f) ** 2 * finite.to(sim.dtype)
        n_valid = finite.to(sim.dtype).sum(dim=1).clamp(min=1.0)
        overall = torch.sqrt(sq.sum(dim=1) / n_valid + self.eps)

        obs_for_peak = obs.masked_fill(~finite, float('-inf'))
        peaks = torch.amax(obs_for_peak, dim=1)
        finite_peak = torch.isfinite(peaks)
        thresh = self.peak_ratio * peaks
        high_mask = (obs >= thresh.unsqueeze(1)) & finite & finite_peak.unsqueeze(1)
        has_high = high_mask.any(dim=1)
        win_n = high_mask.to(sim.dtype).sum(dim=1).clamp(min=1.0)
        win_mse = (sq * high_mask.to(sim.dtype)).sum(dim=1) / win_n
        win = torch.sqrt(win_mse + self.eps)
        win = torch.where(has_high, win, overall)

        sample = self.overall_weight * overall + self.win_weight * win
        ok = finite.any(dim=1) & finite_peak & torch.isfinite(sample)
        if not ok.any():
            return zero_train_loss(output)
        return sample[ok].mean()


LOSS_FUNCTIONS = {
    'RMSE': RMSELoss,
    'PF_PEAK80': PFPeak80Loss,
}
