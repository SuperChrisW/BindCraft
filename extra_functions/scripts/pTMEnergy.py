# Implementation of inter-chain pTMEnergy loss for colabdesign
# Wang Liyao, 2025-06-03

import jax
import jax.numpy as jnp
import numpy as np
import scipy
from colabdesign.af.alphafold.common import confidence

# place the following functions (LogSumExp, compute_pTMEnergy_score) into colabdesign.af.alphafold.common.confidence
def LogSumExp(logits, axis=None, keepdims=False, use_jnp=False):
    """Computes the log-sum-exp of logits."""
    if use_jnp:
        return jax.scipy.special.logsumexp(logits, axis=axis, keepdims=keepdims)
    else:
        return scipy.special.logsumexp(logits, axis=axis, keepdims=keepdims)

def compute_pTMEnergy_score(logits, breaks, residue_weights=None,
    asym_id=None, use_jnp=False):
    """Computes predicted TM energy score.

    Args:
    logits: [num_res, num_res, num_bins] the logits output from
    PredictedAlignedErrorHead.
    breaks: [num_bins] the error bins.
    residue_weights: [num_res] the per residue weights to use for the
    expectation.
    asym_id: [num_res] the asymmetric unit ID - the chain ID. Only needed for
    ipTM calculation.

    Returns:
    ptmEnergy_score: The predicted TM energy score.
    """
    if use_jnp:
        _np, _softmax = jnp, jax.nn.softmax
    else:
        _np, _softmax = np, scipy.special.softmax

    # residue_weights has to be in [0, 1], but can be floating-point, i.e. the
    # exp. resolved head's probability.
    if residue_weights is None:
        residue_weights = _np.ones(logits.shape[0])

    bin_centers = confidence._calculate_bin_centers(breaks, use_jnp=use_jnp)
    num_res = residue_weights.shape[0]

    # Clip num_res to avoid negative/undefined d0.
    clipped_num_res = _np.maximum(residue_weights.sum(), 19)

    # Compute d_0(num_res) as defined by TM-score, eqn. (5) in Yang & Skolnick
    # "Scoring function for automated assessment of protein structure template
    # quality", 2004: http://zhanglab.ccmb.med.umich.edu/papers/2004_3.pdf
    d0 = 1.24 * (clipped_num_res - 15) ** (1./3) - 1.8

    # TM-Score term for every bin.
    tm_per_bin = 1. / (1 + _np.square(bin_centers) / _np.square(d0))

    predicted_tm_Energy = LogSumExp(logits * tm_per_bin, axis=-1,keepdims=False, use_jnp=use_jnp)

    if asym_id is None:
        pair_mask = _np.full((num_res,num_res),True)
    else:
        pair_mask = asym_id[:, None] != asym_id[None, :]

    predicted_tm_Energy *= pair_mask

    pair_residue_weights = pair_mask * (residue_weights[None, :] * residue_weights[:, None])
    normed_residue_mask = pair_residue_weights / (1e-8 + pair_residue_weights.sum(-1, keepdims=True))
    return (-predicted_tm_Energy * normed_residue_mask).sum(-1).mean()

# place into colabdesign.af.loss
def get_pTMEnergy(inputs, outputs, interface=False):
    pae = {"residue_weights":inputs["seq_mask"],
            **outputs["predicted_aligned_error"]}
    if interface:
        if "asym_id" not in pae:
            pae["asym_id"] = inputs["asym_id"]
    else:
        if "asym_id" in pae:
            pae.pop("asym_id")
    return confidence.compute_pTMEnergy_score(**pae, use_jnp=True)

# place into colabdesign_utils
def add_pTMEnergy_loss(self, weight=0.1):
    def loss_pTMEnergy(inputs, outputs):
        p = get_pTMEnergy(inputs, outputs, interface=True)
        return {"ptmEnergy": mask_loss(p)}
    
    self._callbacks["model"]["loss"].append(loss_pTMEnergy)
    self.opt["weights"]["ptmEnergy"] = weight