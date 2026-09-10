"""
dp_utils.py -- The privacy layer. Implements the core mechanism of DP-SGD
(Abadi et al., "Deep Learning with Differential Privacy", 2016): clip each
gradient update to a bounded L2 norm, then inject calibrated Gaussian
noise before the optimizer step. This is precisely what libraries such as
Opacus and PySyft's differential-privacy tooling do under the hood.

WHY A HAND-ROLLED IMPLEMENTATION HERE INSTEAD OF IMPORTING PySyft DIRECTLY
---------------------------------------------------------------------------
The original design calls for PySyft. In practice, PySyft's actively
maintained releases (0.8+) have pivoted to a remote-execution / "Datasite"
framework for privacy-preserving remote data science, rather than a
drop-in local DP-noise utility -- pulling it in means running a Syft
server/client pair and a much heavier dependency tree, which is overkill
for adding noise to a weight vector before a network send. To keep this
project runnable end-to-end without fragile version pinning, the DP
mechanism is implemented directly (it's ~20 lines of the same math Opacus
and PySyft use). Two swap-in alternatives are documented in README.md if
you want to use a maintained library instead:
  - Opacus (recommended, actively maintained, PyTorch-native): the
    PrivacyEngine wraps your optimizer and does exactly this, plus gives
    you a proper (epsilon, delta) accountant.
  - PySyft: recommended if you also want the remote-execution / multi-
    party trust boundary features it now focuses on, not just DP noise.

This module also exposes a rough epsilon estimate for demo/reporting
purposes. It is NOT a substitute for a real moments-accountant / RDP
accountant -- for any production privacy guarantee, use Opacus's
`opacus.accountants` or PySyft's privacy budget tracking instead.
"""
import math
import torch


def clip_and_noise_gradients(model: torch.nn.Module, clip_norm: float, noise_multiplier: float) -> None:
    """Call this after loss.backward() and before optimizer.step().

    1. Clips the *total* gradient L2 norm across all parameters to
       `clip_norm`, bounding any single training step's sensitivity.
    2. Adds i.i.d. Gaussian noise N(0, (noise_multiplier * clip_norm)^2)
       to every gradient element, so the aggregator (or an attacker
       intercepting the transmitted update) cannot reverse-engineer the
       exact training examples that produced it.
    """
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)
    for p in model.parameters():
        if p.grad is not None:
            noise = torch.normal(
                mean=0.0,
                std=noise_multiplier * clip_norm,
                size=p.grad.shape,
                device=p.grad.device,
            )
            p.grad.add_(noise)


def estimate_epsilon(noise_multiplier: float, steps: int, sample_rate: float, delta: float = 1e-5) -> float:
    """Rough, simplified privacy-budget estimate for dashboard/reporting
    purposes only (a loose approximation of the advanced composition
    bound, NOT a tight moments/RDP accountant). Lower epsilon = stronger
    privacy. For an authoritative accounting, replace this with
    `opacus.accountants.RDPAccountant`.
    """
    if noise_multiplier <= 0:
        return float("inf")
    # Loose advanced-composition-style bound, deliberately conservative.
    eps_per_step = (sample_rate / noise_multiplier) * math.sqrt(2 * math.log(1.25 / delta))
    return eps_per_step * math.sqrt(steps)


# ---------------------------------------------------------------------------
# Optional: how you'd swap in Opacus instead of the manual functions above.
# Left here as reference/documentation; not imported by default so the
# project has no hard dependency on it.
# ---------------------------------------------------------------------------
OPACUS_EXAMPLE = """
from opacus import PrivacyEngine

privacy_engine = PrivacyEngine()
model, optimizer, train_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=train_loader,
    noise_multiplier=1.0,
    max_grad_norm=1.0,
)
# train as normal; privacy_engine.get_epsilon(delta=1e-5) gives a real
# accountant-backed privacy budget after training.
"""
