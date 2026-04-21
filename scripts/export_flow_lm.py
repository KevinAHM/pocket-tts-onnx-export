import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path
import onnx
import onnxruntime as ort
import numpy as np
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_LANGUAGE
from pocket_tts.modules.stateful_module import init_states, StatefulModule
from pocket_tts.modules.transformer import StreamingMultiheadAttention
from pocket_tts.modules.mimi_transformer import MimiStreamingMultiheadAttention, KVCacheResult
from onnx_export.bundle_metadata import write_bundle_metadata
from onnx_export.export_utils import get_state_structure, flatten_state

# ==============================================================================
# 1. MONKEYPATCHES
# ==============================================================================

def patched_init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
    dim_per_head = self.embed_dim // self.num_heads
    initial_step = torch.tensor([0], dtype=torch.long, device=self.in_proj.weight.device)
    initial_current_end = torch.zeros((0,)).to(self.in_proj.weight.device)
    return dict(
        step=initial_step,
        current_end=initial_current_end,
        cache=torch.full(
            (2, batch_size, sequence_length, self.num_heads, dim_per_head),
            float("NaN"),
            device=self.in_proj.weight.device,
            dtype=self.in_proj.weight.dtype,
        ),
    )

def patched_increment_step(self, state: dict, increment: int = 1):
    state["step"] = state["step"] + increment

def patched_streaming_offset(self, state: dict | None) -> torch.Tensor:
    return state["step"]

def patched_sma_complete_kv(self, k, v, state: dict | None):
    current_step = state["step"]
    cache = state["cache"]
    new_cache = cache.clone()
    new_cache[0, :, current_step : current_step + k.shape[1]] = k
    new_cache[1, :, current_step : current_step + v.shape[1]] = v
    state["cache"] = new_cache
    valid = new_cache[:, :, : current_step + k.shape[1]]
    return valid[0], valid[1]

def patched_get_mask(self, shape: tuple[int, torch.Tensor], shift: torch.Tensor, device: torch.device):
    rows, cols_tensor = shape
    row_idx = torch.arange(rows, device=device).unsqueeze(1) 
    MAX_COLS = 4096 
    full_col_idx = torch.arange(MAX_COLS, device=device).unsqueeze(0)
    col_idx = full_col_idx[:, :cols_tensor]
    mask_bool = (col_idx <= row_idx + shift)
    mask = torch.full(mask_bool.shape, float("-inf"), device=device)
    mask.masked_fill_(mask_bool, 0.0)
    return mask

def patched_sma_forward(self, query: torch.Tensor, model_state: dict | None):
    state = self.check_model_state(model_state)
    projected = self.in_proj(query)
    b, t, _ = projected.shape
    d = self.embed_dim // self.num_heads
    packed = projected.view(b, t, 3, self.num_heads, d)
    q, k, v = torch.unbind(packed, dim=2)
    q, k = self._apply_rope(q, k, state)
    k, v = self._complete_kv(k, v, state)

    current_step = state["step"]
    mask_shape = (t, t + current_step)
    shift = current_step
    attn_mask = self._get_mask(mask_shape, shift=shift, device=q.device)

    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    x = F.scaled_dot_product_attention(q, k, v, attn_mask)
    x = x.transpose(1, 2)
    x = x.reshape(b, t, self.num_heads * d)
    x = self.out_proj(x)
    return x

StreamingMultiheadAttention.init_state = patched_init_state
StreamingMultiheadAttention.increment_step = patched_increment_step
StreamingMultiheadAttention._streaming_offset = patched_streaming_offset
StreamingMultiheadAttention._complete_kv = patched_sma_complete_kv
StreamingMultiheadAttention._get_mask = patched_get_mask
StreamingMultiheadAttention.forward = patched_sma_forward

def patched_mimi_increment_step(self, state: dict, increment: int = 1):
    state["offset"] = state["offset"] + increment

MimiStreamingMultiheadAttention.increment_step = patched_mimi_increment_step

def patched_stateful_increment_step(self, state: dict, increment = 1):
    return state
StatefulModule.increment_step = patched_stateful_increment_step

# ==============================================================================
# 2. WRAPPERS
# ==============================================================================

class FlowLMMainWrapper(nn.Module):
    """
    Unified Backbone Model for both Conditioning and AR steps.
    Inputs: 
      - sequence: (B, T, 32)
      - text_embeddings: (B, Text, 1024)
      - state_*: KVCache states
    Outputs:
      - conditioning: (B, 1024) - used for Flow step
      - eos_logit: (B, 1) - used for EOS detection
      - out_state_*: Updated states
    """
    def __init__(self, flow_lm, state_structure, eos_threshold=-4.0):
        super().__init__()
        self.flow_lm = flow_lm
        self.state_structure = state_structure
        self.eos_threshold = eos_threshold
        
    def forward(self, sequence, text_embeddings, state_flat):
        idx = 0
        def unflatten_recursive(struct):
            nonlocal idx
            s = {}
            for k, v in  sorted(struct.items()):
                if isinstance(v, dict): e = unflatten_recursive(v)
                else: 
                    e = state_flat[idx]
                    idx += 1
                s[k] = e
            return s
        model_state = unflatten_recursive(self.state_structure)
        
        # Handle BOS replacement (NaN -> bos_emb)
        sequence = torch.where(torch.isnan(sequence), self.flow_lm.bos_emb, sequence)

        input_ = self.flow_lm.input_linear(sequence)
        
        # Backbone Forward Pass
        # Returns (B, T_new, 1024)
        transformer_out = self.flow_lm.backbone(input_, text_embeddings, sequence, model_state=model_state)
        
        # Extract Conditioning
        batch_size = transformer_out.shape[0]
        dim = transformer_out.shape[2]
        dummy_out = torch.zeros((batch_size, 1, dim), device=transformer_out.device, dtype=transformer_out.dtype)
        
        # We use [(transformer_out + dummy)]
        augmented_out = torch.cat([transformer_out, dummy_out], dim=1)
        c = augmented_out[:, 0]
        
        # Extract EOS logit
        eos_logit = self.flow_lm.out_eos(c)

        # State Increment
        seq_len = sequence.shape[1]
        text_len = text_embeddings.shape[1]
        increment = seq_len + text_len
        
        def recurse_increment(s):
            if "step" in s: s["step"] = s["step"] + increment
            if "offset" in s: s["offset"] = s["offset"] + increment
            for k, v in s.items(): 
                if isinstance(v, dict): recurse_increment(v)
        recurse_increment(model_state)
        
        from onnx_export.export_utils import flatten_state as fs
        out_state = fs(model_state)
        
        return c, eos_logit, *out_state


class FlowNetWrapper(nn.Module):
    """
    Stateless wrapper for FlowNet.
    Inputs: c (conditioning), s (timestep), t (timestep), x (latent)
    Output: flow_dir
    """
    def __init__(self, flow_lm):
        super().__init__()
        self.flow_net = flow_lm.flow_net
        
    def forward(self, c, s, t, x):
        return self.flow_net(c, s, t, x)


def compare_outputs(name, torch_output, onnx_output, exact: bool):
    if exact:
        if np.array_equal(torch_output, onnx_output):
            print(f"{name} matches exactly.")
            return
        diff = np.abs(torch_output.astype(np.float64) - onnx_output.astype(np.float64))
        raise AssertionError(
            f"{name} mismatch. max_abs_diff={diff.max()} mean_abs_diff={diff.mean()}"
        )

    np.testing.assert_allclose(torch_output, onnx_output, rtol=2e-5, atol=2e-5)
    print(f"{name} matches within tolerance.")


def verify_export(tts, structure, flat_state, main_out_path, flow_out_path, exact: bool):
    print("\nVerifying FlowLM exports...")
    ort_main = ort.InferenceSession(main_out_path)
    ort_flow = ort.InferenceSession(flow_out_path)

    main_wrapper = FlowLMMainWrapper(tts.flow_lm, structure)
    flow_wrapper = FlowNetWrapper(tts.flow_lm)

    test_seq = torch.randn(1, 1, tts.flow_lm.ldim)
    test_text = torch.randn(1, 3, tts.flow_lm.dim)

    with torch.no_grad():
        pt_main = main_wrapper(test_seq, test_text, flat_state)

    ort_inputs = {
        "sequence": test_seq.numpy(),
        "text_embeddings": test_text.numpy(),
    }
    for i, state_tensor in enumerate(flat_state):
        ort_inputs[f"state_{i}"] = state_tensor.numpy()
    onnx_main = ort_main.run(None, ort_inputs)

    compare_outputs("FlowLM conditioning", pt_main[0].numpy(), onnx_main[0], exact)
    compare_outputs("FlowLM eos_logit", pt_main[1].numpy(), onnx_main[1], exact)
    for i, (pt_state, onnx_state) in enumerate(zip(pt_main[2:], onnx_main[2:])):
        compare_outputs(f"FlowLM state {i}", pt_state.numpy(), onnx_state, exact)

    test_c = torch.randn(1, tts.flow_lm.dim)
    test_s = torch.tensor([[0.0]], dtype=torch.float32)
    test_t = torch.tensor([[1.0]], dtype=torch.float32)
    test_x = torch.randn(1, tts.flow_lm.ldim)

    with torch.no_grad():
        pt_flow = flow_wrapper(test_c, test_s, test_t, test_x)
    onnx_flow = ort_flow.run(
        None,
        {"c": test_c.numpy(), "s": test_s.numpy(), "t": test_t.numpy(), "x": test_x.numpy()},
    )[0]
    compare_outputs("Flow net output", pt_flow.numpy(), onnx_flow, exact)
    print("FlowLM verification successful.")


# ==============================================================================
# 3. EXPORT SCRIPT
# ==============================================================================

def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Export FlowLM models to ONNX.")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_models", help="Directory for output ONNX files")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="Model language/config name.")
    parser.add_argument("--config", type=str, default=None, help="Path to a local YAML config file.")
    parser.add_argument("--exact", action="store_true", help="Require exact torch vs ONNX equality.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading model...")
    tts = TTSModel.load_model(language=args.language, config=args.config).cpu().eval()
    bundle_name = Path(args.config).stem if args.config is not None else args.language
            
    # Init patched state
    STATIC_SEQ_LEN = 1000
    state = init_states(tts.flow_lm, batch_size=1, sequence_length=STATIC_SEQ_LEN)
    structure = get_state_structure(state)
    flat_state = flatten_state(state)
    write_bundle_metadata(args.output_dir, tts, bundle_name=bundle_name, flow_state=state)
    
    state_input_names = [f"state_{i}" for i in range(len(flat_state))]
    state_output_names = [f"out_state_{i}" for i in range(len(flat_state))]
    
    # -------------------------------------------------------------
    # 1. Main Flow Model (Backbone)
    print("\nExporting FlowLM Main Model (Backbone)...")
    main_wrapper = FlowLMMainWrapper(tts.flow_lm, structure)
    
    # Needs to handle dynamic axes for both seq and text
    dummy_seq = torch.randn(1, 1, tts.flow_lm.ldim)
    dummy_text = torch.randn(1, 1, tts.flow_lm.dim)
    main_args = (dummy_seq, dummy_text, flat_state)
    
    main_out_path = os.path.join(args.output_dir, "flow_lm_main.onnx")
    torch.onnx.export(
        main_wrapper, main_args, main_out_path,
        input_names=["sequence", "text_embeddings"] + state_input_names,
        output_names=["conditioning", "eos_logit"] + state_output_names,
        dynamic_axes={"sequence": {1: "seq_len"}, "text_embeddings": {1: "text_len"}},
        opset_version=14, dynamo=False
    )
    print(f"Exported {main_out_path}")
    
    # 2. Flow Net Model
    print("\nExporting Flow Net Model...")
    flow_wrapper = FlowNetWrapper(tts.flow_lm)
    # Inputs: c(B, 1024), s(B,1), t(B,1), x(B, 32)
    dummy_c = torch.randn(1, 1024)
    dummy_s = torch.tensor([[0.0]])
    dummy_t = torch.tensor([[1.0]])
    dummy_x = torch.randn(1, 32)
    
    flow_args = (dummy_c, dummy_s, dummy_t, dummy_x)
    
    flow_out_path = os.path.join(args.output_dir, "flow_lm_flow.onnx")
    torch.onnx.export(
        flow_wrapper, flow_args, flow_out_path,
        input_names=["c", "s", "t", "x"],
        output_names=["flow_dir"],
        dynamic_axes={"c": {0: "batch"}, "s": {0: "batch"}, "t": {0: "batch"}, "x": {0: "batch"}},
        opset_version=14, dynamo=False
    )
    print(f"Exported {flow_out_path}")
    verify_export(tts, structure, flat_state, main_out_path, flow_out_path, exact=args.exact)
    print("\nDone! 2-Model split optimization complete.")

if __name__ == "__main__":
    main()
