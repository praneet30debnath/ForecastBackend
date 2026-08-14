# forecast_service/attention.py
#
# Runs the same autoregressive generation KronosPredictor.predict() does, but
# temporarily swaps each layer's self-attention forward pass for a version
# that computes attention weights explicitly (instead of the fused kernel,
# which doesn't expose them) and records the query row for the token being
# generated at each step. Averaging those rows tells us which historical
# days the model leaned on most while producing this forecast.
#
# Caveat inherited by callers: attention weight is a heuristic proxy for
# influence, not a rigorous causal explanation of the prediction.
import math
import types

import numpy as np
import pandas as pd
import torch

from model.kronos import auto_regressive_inference, calc_time_stamps


def _capturing_forward(self, x, key_padding_mask=None, _sink=None):
    batch_size, seq_len, _ = x.shape
    q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
    q, k = self.rotary(q, k)

    scale = 1.0 / math.sqrt(self.head_dim)
    scores = (q @ k.transpose(-2, -1)) * scale
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool), diagonal=1
    )
    scores = scores.masked_fill(causal_mask, float("-inf"))
    weights = torch.softmax(scores, dim=-1)

    # Only the newest position's attention row explains the token about to
    # be generated; average over heads now, over layers/steps at call site.
    _sink.append(weights[:, :, -1, :].mean(dim=1).detach().cpu().numpy())

    attn_output = weights @ v
    attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
    return self.resid_dropout(self.out_proj(attn_output))


def predict_with_attention(predictor, df, x_timestamp, y_timestamp, pred_len, T=1.0, top_p=0.9):
    price_cols = predictor.price_cols
    vol_col, amt_col = predictor.vol_col, predictor.amt_vol

    df = df.copy()
    if vol_col not in df.columns:
        df[vol_col] = 0.0
        df[amt_col] = 0.0
    if amt_col not in df.columns:
        df[amt_col] = df[vol_col] * df[price_cols].mean(axis=1)

    x_stamp = calc_time_stamps(x_timestamp).values.astype(np.float32)
    y_stamp = calc_time_stamps(y_timestamp).values.astype(np.float32)
    x = df[price_cols + [vol_col, amt_col]].values.astype(np.float32)

    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
    x_norm = np.clip((x - x_mean) / (x_std + 1e-5), -predictor.clip, predictor.clip)

    x_tensor = torch.from_numpy(x_norm[np.newaxis, :]).to(predictor.device)
    x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis, :]).to(predictor.device)
    y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis, :]).to(predictor.device)

    initial_seq_len = x_tensor.size(1)
    n_layers = len(predictor.model.transformer)
    captured = []

    originals = [block.self_attn.forward for block in predictor.model.transformer]
    for block in predictor.model.transformer:
        block.self_attn.forward = types.MethodType(
            lambda self, x, key_padding_mask=None, _sink=captured: _capturing_forward(
                self, x, key_padding_mask, _sink
            ),
            block.self_attn,
        )
    try:
        preds = auto_regressive_inference(
            predictor.tokenizer, predictor.model,
            x_tensor, x_stamp_tensor, y_stamp_tensor,
            predictor.max_context, pred_len, predictor.clip,
            T, 0, top_p, 1, False,
        )
    finally:
        for block, orig in zip(predictor.model.transformer, originals):
            block.self_attn.forward = orig

    preds = preds[:, -pred_len:, :].squeeze(0) * (x_std + 1e-5) + x_mean
    pred_df = pd.DataFrame(preds, columns=price_cols + [vol_col, amt_col], index=y_timestamp)

    # `captured` has n_layers entries per generation step, oldest first.
    # Each entry has shape [1, seq_len_at_that_step]; seq_len grows by one
    # per step. Average across layers, then across steps, restricted to the
    # historical portion of the window (indices < initial_seq_len).
    per_date_weight = np.zeros(initial_seq_len, dtype=np.float64)
    historical_reliance = []
    for step in range(pred_len):
        layer_rows = captured[step * n_layers:(step + 1) * n_layers]
        step_avg = np.mean([row[0] for row in layer_rows], axis=0)
        hist_part = step_avg[:initial_seq_len]
        historical_reliance.append(float(hist_part.sum()))  # vs. self-generated future days
        per_date_weight += hist_part / (hist_part.sum() + 1e-9)
    per_date_weight /= pred_len

    return pred_df, per_date_weight, float(np.mean(historical_reliance))
