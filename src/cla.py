"""Cross-LoRA Attention (CLA).

Per-layer wrapper of LlamaMLP that fuses N per-doc LoRAs through a learned
cross-attention + token-level router.

Two unit-strength injections (no weights — bias terms added at full strength):

  cross_attn_logits += log(R)              # doc-doc complementarity prior
  router_logits     += log_softmax(scores) # query-doc relevance prior

R is the (N, N) query-conditioned complementarity matrix; scores are the (N,)
per-doc ColBERT MaxSim totals. Both come from one ColBERT pass over query+docs.

  H_tilde = H1 + alpha * cross_attn_out
  output  = base_mlp(x) + router(H_tilde, lora_gate, lora_down)

Modules:
  - CLAModule:    cross-attn + router projections; alpha scalar (frozen, from config).
  - CLAMLP:       per-layer wrapper of base LlamaMLP.
  - MountedLoRAs: context manager attaching N doc-LoRAs + R + scores to the module.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.lora import load_doc_lora_on_device


# ============================================================================
# CLAModule: trainable params + runtime context
# ============================================================================

class CLAModule(nn.Module):
    """Container for CLA trainable parameters + runtime context.

    Args:
        hidden_size: base LLM's hidden dim (d). 2048 for llama-3.2-1b.
        intermediate_size: base LLM's MLP intermediate dim (d_m). 8192 for llama-3.2-1b.
        L_cla: list of layer indices to mount CLA on.
        d_a: per-layer cross-attention internal dim. 64 by default.
        alpha: scalar multiplier on the cross-attn residual contribution.
               Tuned per-deployment; see docs/EXPERIMENT_LOG.md.
    """

    def __init__(self, hidden_size, intermediate_size, L_cla, d_a=64, alpha=0.1):
        super().__init__()
        self.d = hidden_size
        self.d_m = intermediate_size
        self.d_a = d_a
        self.L_cla = sorted(set(L_cla))
        self.alpha = float(alpha)

        self.layer_params = nn.ModuleDict()
        for l in self.L_cla:
            self.layer_params[str(l)] = nn.ModuleDict({
                'W_Q_cross': nn.Linear(self.d_m, self.d_a, bias=False),
                'W_K_cross': nn.Linear(self.d_m, self.d_a, bias=False),
                'W_V_cross': nn.Linear(self.d_m, self.d_a, bias=False),
                'W_O_cross': nn.Linear(self.d_a, self.d_m, bias=False),
                'W_Q_route': nn.Linear(self.d, self.d_a, bias=False),
                'W_K_route': nn.Linear(self.d_m, self.d_a, bias=False),
            })

        # Runtime context, set via MountedLoRAs. Not persisted in state_dict.
        self._loras = None
        self._R = None
        self._scores = None

    def set_context(self, loras, R, scores):
        self._loras = loras
        self._R = R
        self._scores = scores

    def clear_context(self):
        self._loras = None
        self._R = None
        self._scores = None

    @property
    def has_context(self):
        return self._loras is not None and self._R is not None and self._scores is not None


# ============================================================================
# CLAMLP: per-layer wrapper of base LlamaMLP
# ============================================================================

class CLAMLP(nn.Module):
    """Wraps a single base LlamaMLP at `layer_idx`.

    Falls through to the base MLP when the CLAModule has no context.
    """

    def __init__(self, base_mlp, layer_idx, cla_module):
        super().__init__()
        self.base_mlp = base_mlp
        self.layer_idx = layer_idx
        self.cla_module = cla_module

    def forward(self, x):
        if not self.cla_module.has_context:
            return self.base_mlp(x)

        N = len(self.cla_module._loras)
        if N == 0:
            return self.base_mlp(x)

        loras = []
        for n in range(N):
            adapter = self.cla_module._loras[n]
            l_data = adapter['by_layer'].get(self.layer_idx)
            if l_data is None or any(p not in l_data for p in ('gate', 'up', 'down')):
                raise ValueError(f'adapter {n} missing layer {self.layer_idx} weights for gate/up/down')
            loras.append({'gate': l_data['gate'], 'up': l_data['up'], 'down': l_data['down'], 'scaling': adapter['scaling']})

        R = self.cla_module._R
        params = self.cla_module.layer_params[str(self.layer_idx)]
        d_a = self.cla_module.d_a
        alpha = self.cla_module.alpha

        # Ablation toggles (env-var, read each forward call so they can be set after model load)
        import os as _os
        NO_CROSS  = _os.environ.get('CLA_NO_CROSS',  '0') == '1'
        NO_ROUTER = _os.environ.get('CLA_NO_ROUTER', '0') == '1'
        NO_R      = _os.environ.get('CLA_NO_R',      '0') == '1'
        NO_C      = _os.environ.get('CLA_NO_C',      '0') == '1'

        # === Base SwiGLU components ===
        base_gate = self.base_mlp.gate_proj(x)
        base_up = self.base_mlp.up_proj(x)

        # === Step 1: per-LoRA up_proj activations h1_n ===
        h1_list = []
        for n in range(N):
            A_up, B_up = loras[n]['up']
            scaling = loras[n]['scaling']
            h1 = (x @ A_up.T) @ B_up.T * scaling
            h1_list.append(h1)
        H1 = torch.stack(h1_list, dim=-2)              # (b, s, N, d_m)

        H1_fp32 = H1.float()

        # === Step 2-3: Cross-LoRA Attention + residual ===
        if NO_CROSS:
            H_tilde = H1
        else:
            Q = params['W_Q_cross'](H1_fp32)
            K = params['W_K_cross'](H1_fp32)
            V = params['W_V_cross'](H1_fp32)
            scores = (Q @ K.transpose(-1, -2)) / math.sqrt(d_a)
            if not NO_R:
                log_R = torch.log(R.clamp(min=1e-8)).to(scores.dtype).to(scores.device)
                scores = scores + log_R
            beta = F.softmax(scores, dim=-1)
            attn_out = beta @ V
            attn_proj_dm = params['W_O_cross'](attn_out)
            H_tilde = H1 + (alpha * attn_proj_dm).to(H1.dtype)

        # === Step 4: Token-level router with log_softmax(scores) bias ===
        if NO_ROUTER:
            b_, s_ = H1.shape[0], H1.shape[1]
            router_alpha = torch.full((b_, s_, N), 1.0 / N, dtype=H1.dtype, device=H1.device)
        else:
            q_router_fp32 = params['W_Q_route'](x.float())
            k_router_fp32 = params['W_K_route'](H1_fp32)
            alpha_logits = (q_router_fp32.unsqueeze(-2) * k_router_fp32).sum(-1) / math.sqrt(d_a)
            if not NO_C:
                log_prior = F.log_softmax(self.cla_module._scores.float().to(alpha_logits.device), dim=-1)
                alpha_logits = alpha_logits + log_prior.view(1, 1, -1).to(alpha_logits.dtype)
            router_alpha = F.softmax(alpha_logits, dim=-1).to(H1.dtype)

        # === Step 5: SwiGLU integration ===
        h_lora_up = (router_alpha.unsqueeze(-1) * H_tilde).sum(dim=-2)

        h_lora_gate = torch.zeros_like(base_gate)
        for n in range(N):
            A_g, B_g = loras[n]['gate']
            scaling = loras[n]['scaling']
            h_g = (x @ A_g.T) @ B_g.T * scaling
            h_lora_gate = h_lora_gate + router_alpha[..., n:n+1] * h_g

        gate_out = F.silu(base_gate + h_lora_gate)
        up_out = base_up + h_lora_up
        h_mid = gate_out * up_out

        base_down = self.base_mlp.down_proj(h_mid)
        h_lora_down = torch.zeros_like(base_down)
        for n in range(N):
            A_d, B_d = loras[n]['down']
            scaling = loras[n]['scaling']
            h_d = (h_mid @ A_d.T) @ B_d.T * scaling
            h_lora_down = h_lora_down + router_alpha[..., n:n+1] * h_d

        return base_down + h_lora_down


# ============================================================================
# Mount / unmount / context manager
# ============================================================================

def mount_cla(model, cla_module, L_cla):
    """Replace each LlamaMLP at layers in L_cla with a CLAMLP wrapper."""
    layers_path = model.model.layers if hasattr(model, 'model') else model.layers
    for l in sorted(set(L_cla)):
        if isinstance(layers_path[l].mlp, CLAMLP):
            continue
        original_mlp = layers_path[l].mlp
        layers_path[l].mlp = CLAMLP(original_mlp, l, cla_module)
    print(f'CLA mounted on layers {sorted(set(L_cla))}')


def unmount_cla(model):
    """Restore each LlamaMLP that was replaced by CLAMLP."""
    layers_path = model.model.layers if hasattr(model, 'model') else model.layers
    for l in range(len(layers_path)):
        mod = layers_path[l].mlp
        if isinstance(mod, CLAMLP):
            layers_path[l].mlp = mod.base_mlp


class MountedLoRAs:
    """Context manager: attach N doc-LoRAs + R + scores to a CLA-mounted model.

    Usage:
        with MountedLoRAs(cla_module, lora_paths, R, scores, device='cuda', dtype=torch.bfloat16):
            output = model(input_ids)

    R:      (N, N) float tensor — query-conditioned doc-doc complementarity.
    scores: (N,) float tensor   — per-doc ColBERT query relevance.
    """

    def __init__(self, cla_module, lora_paths, R, scores, device='cuda', dtype=torch.bfloat16):
        self.cla_module = cla_module
        self.lora_paths = list(lora_paths)
        self.R = R
        self.scores = scores
        self.device = device
        self.dtype = dtype

    def __enter__(self):
        N = len(self.lora_paths)
        R_t = (self.R if torch.is_tensor(self.R) else torch.tensor(self.R, dtype=torch.float32)).float()
        s_t = (self.scores if torch.is_tensor(self.scores) else torch.tensor(self.scores, dtype=torch.float32)).float()
        assert R_t.shape == (N, N), f'R shape {tuple(R_t.shape)} != ({N}, {N})'
        assert s_t.shape == (N,), f'scores shape {tuple(s_t.shape)} != ({N},)'
        R_t = R_t.to(self.device); s_t = s_t.to(self.device)
        loras = [load_doc_lora_on_device(path, self.device, self.dtype) for path in self.lora_paths]
        self.cla_module.set_context(loras, R_t, s_t)
        return self

    def __exit__(self, *args):
        self.cla_module.clear_context()
