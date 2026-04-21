
import sys
import argparse

# Monkeypatch beartype to avoid errors during tracing where Tensors are passed as ints
import beartype
beartype.beartype = lambda *args, **kwargs: (lambda func: func) if not args else args[0]

import torch
# Monkeypatch trunc_normal_ to be ONNX-friendly
def patched_trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # Use normal distribution clamped to range.
    # This avoids generic aten::uniform or complex rejection sampling loops in export
    with torch.no_grad():
        if std == 0:
            return tensor.fill_(mean).clamp_(a, b)
        return tensor.normal_(mean, std).clamp_(a, b)

torch.nn.init.trunc_normal_ = patched_trunc_normal_

# Monkeypatch global increment_steps to update scalar 'step' instead of resizing tensor
import pocket_tts.modules.stateful_module as stateful_module
from pocket_tts.modules.stateful_module import StatefulModule

def patched_increment_steps(module, model_state, increment=1):
    for module_name, m in module.named_modules():
        if not isinstance(m, StatefulModule):
            continue
        # Call patched increment_step on module
        m.increment_step(model_state[module_name], increment)

stateful_module.increment_steps = patched_increment_steps

# Monkeypatch StatefulModule.increment_step to allow Tensors and avoid beartype issues
def patched_stateful_increment_step(self, state: dict, increment = 1):
    pass
StatefulModule.increment_step = patched_stateful_increment_step

# Monkeypatch StreamingMultiheadAttention to use scalar 'step'
from pocket_tts.modules.transformer import StreamingMultiheadAttention, complete_kv

def patched_init_state(self, batch_size: int, sequence_length: int) -> dict[str, torch.Tensor]:
    dim_per_head = self.embed_dim // self.num_heads
    # Use a scalar tensor 'step' for position tracking
    initial_step = torch.tensor([0], dtype=torch.long, device=self.in_proj.weight.device)
    # current_end is kept static (size 0) or dummy, we won't use it for length info
    initial_current_end = torch.zeros((0,)).to(self.in_proj.weight.device)
    
    return dict(
        step=initial_step,
        current_end=initial_current_end, # Kept for compatibility if accessed elsewhere
        cache=torch.full(
            (2, batch_size, sequence_length, self.num_heads, dim_per_head),
            float("NaN"),
            device=self.in_proj.weight.device,
            dtype=self.in_proj.weight.dtype,
        ),
    )

def patched_increment_step(self, state: dict, increment: int = 1):
    # Update scalar step
    state["step"] = state["step"] + increment

def patched_streaming_offset(self, state: dict | None) -> torch.Tensor:
    # Read position from scalar step as TENSOR
    return state["step"]

def patched_sma_complete_kv(self, k, v, state: dict | None):
    # current_step IS TENSOR
    current_step = state["step"]
    
    # Clone cache for out-of-place update (ONNX requirement)
    cache = state["cache"]
    new_cache = cache.clone()
    
    # Slice assignment with tensors should work in modern Torch -> DynamicSlice in ONNX
    new_cache[0, :, current_step : current_step + k.shape[1]] = k
    new_cache[1, :, current_step : current_step + v.shape[1]] = v
    
    # Update state dict
    state["cache"] = new_cache
    
    # Slicing logic from original: valid = cache[:, :, : current_end + k.shape[1]]
    valid = new_cache[:, :, : current_step + k.shape[1]]
    return valid[0], valid[1]

StreamingMultiheadAttention.init_state = patched_init_state
StreamingMultiheadAttention.increment_step = patched_increment_step
StreamingMultiheadAttention._streaming_offset = patched_streaming_offset
StreamingMultiheadAttention._complete_kv = patched_sma_complete_kv

# Monkeypatch _get_mask to use arithmetic implementation (avoids torch.tril specialization)
def patched_get_mask(self, shape: tuple[int, torch.Tensor], shift: torch.Tensor, device: torch.device):
    rows, cols_tensor = shape
    # rows is int (static/symbolic from query), cols_tensor is Tensor (t + step)
    
    # Create row indices [rows, 1]
    row_idx = torch.arange(rows, device=device).unsqueeze(1) 
    
    # Create col indices [1, cols] dynamic size
    # torch.arange(tensor) fails in legacy export.
    # Workaround: Arange larger constant, then slice.
    MAX_COLS = 4096 # Sufficiently large buffer
    
    # We must slice using the tensor.
    # torch.arange(MAX_COLS)[:cols_tensor]
    # Check if cols_tensor is within bounds? If not, we have bigger problems.
    
    full_col_idx = torch.arange(MAX_COLS, device=device).unsqueeze(0)
    col_idx = full_col_idx[:, :cols_tensor]
    
    # Mask condition
    mask_bool = (col_idx <= row_idx + shift)
    
    # Create mask via broadcasting logic
    mask = torch.full(mask_bool.shape, float("-inf"), device=device)
    mask.masked_fill_(mask_bool, 0.0)
    
    return mask

StreamingMultiheadAttention._get_mask = patched_get_mask

# Monkeypatch Forward to use step for mask
import torch.nn.functional as F
def patched_sma_forward(self, query: torch.Tensor, model_state: dict | None):
    # Standard forward logic but using 'step' for shift
    state = self.check_model_state(model_state)

    projected = self.in_proj(query)
    # Reshape from (b, t, p*h*d) to (b, t, p, h, d) where p=3, h=num_heads
    b, t, _ = projected.shape
    # torch._check(t > 0) removed for legacy export compatibility
    
    d = self.embed_dim // self.num_heads
    packed = projected.view(b, t, 3, self.num_heads, d)
    q, k, v = torch.unbind(packed, dim=2)
    q, k = self._apply_rope(q, k, state)
    k, v = self._complete_kv(k, v, state)

    # PATCHED: Use step from state as TENSOR (do not use .item())
    current_step = state["step"]
    
    # Mask shape: (T, current_step + T)
    # Pass as mixed tuple, handler will unpack
    mask_shape = (t, t + current_step)
    shift = current_step

    attn_mask = self._get_mask(mask_shape, shift=shift, device=q.device)

    q, k, v = [x.transpose(1, 2) for x in (q, k, v)]
    x = F.scaled_dot_product_attention(q, k, v, attn_mask)
    x = x.transpose(1, 2)
    x = x.reshape(b, t, self.num_heads * d)
    x = self.out_proj(x)

    return x

StreamingMultiheadAttention.forward = patched_sma_forward

# Monkeypatch MimiStreamingMultiheadAttention logic (mostly same)
from pocket_tts.modules.mimi_transformer import MimiStreamingMultiheadAttention, KVCacheResult

def patched_mimi_increment_step(self, state: dict, increment: int = 1):
    # Corrected: Update "offset" (RoPE pos) not "end_offset" (buffer pos)
    state["offset"] = state["offset"] + increment

MimiStreamingMultiheadAttention.increment_step = patched_mimi_increment_step

# Restore existing Mimi complete_kv patch
def patched_mimi_complete_kv(self, k, v, model_state: dict | None):
    if model_state is None:
        return KVCacheResult.from_kv(k, v)
        
    layer_state = self.get_state(model_state)
    cache = layer_state["cache"]
    end_offset = layer_state["end_offset"]
    
    capacity = cache.shape[3]
    B, H, T, D = k.shape
    
    new_cache = cache.clone()
    new_end_offset = end_offset.clone()
    
    # Original logic adapted...
    indexes = torch.arange(T, device=end_offset.device, dtype=end_offset.dtype)
    indexes = indexes + end_offset.view(-1, 1)
    indexes = indexes % capacity
    
    this_indexes = indexes.view(B, 1, T, 1)
    this_indexes = this_indexes.expand(-1, H, T, D)
    
    new_cache[0].scatter_(2, this_indexes, k)
    new_cache[1].scatter_(2, this_indexes, v)
    
    keys = new_cache[0]
    values = new_cache[1]
    
    indexes_r = torch.arange(capacity, device=end_offset.device, dtype=torch.long)
    last_offset = end_offset.view(-1, 1) + T - 1
    end_index = last_offset % capacity
    delta = indexes_r - end_index
    
    positions = torch.where(delta <= 0, last_offset + delta, last_offset + delta - capacity)
    new_end_offset[:] = end_offset + T
    
    invalid = indexes_r >= new_end_offset.view(-1, 1)
    positions = torch.where(invalid, torch.full_like(positions, -1), positions)
    
    layer_state["cache"] = new_cache
    layer_state["end_offset"] = new_end_offset
    
    return KVCacheResult(keys, values, positions)

MimiStreamingMultiheadAttention._complete_kv = patched_mimi_complete_kv

import os
from pathlib import Path
import onnxruntime as ort
import numpy as np
from pocket_tts.models.tts_model import TTSModel
from pocket_tts.default_parameters import DEFAULT_LANGUAGE
from pocket_tts.modules.stateful_module import init_states
from onnx_export.bundle_metadata import write_bundle_metadata
from onnx_export.export_utils import get_state_structure, flatten_state, unflatten_state

from pocket_tts.modules.conv import StreamingConv1d, StreamingConvTranspose1d
def patched_conv1d_forward(self, x, model_state: dict | None):
    B, C, T = x.shape
    S = self._stride
    # Removed assert for trace
    if model_state is None:
        # Export and inference both use batch size 1, and tracing may present B as a Tensor.
        state = self.init_state(1, 0)
    else:
        state = self.get_state(model_state)
    TP = state["previous"].shape[-1]
    if TP and self.pad_mode == "replicate":
        # assert T >= TP 
        init = x[..., :1]
        # Out-of-place update
        new_prev = torch.where(
            state["first"].view(-1, 1, 1), init, state["previous"]
        )
        state["previous"] = new_prev
    if TP:
        x = torch.cat([state["previous"], x], dim=-1)
    y = self.conv(x)
    if TP:
        # Out-of-place update
        state["previous"] = x[..., -TP:]
        if self.pad_mode == "replicate":
            state["first"] = torch.zeros_like(state["first"])
    return y

def patched_convtr_forward(self, x, mimi_state: dict):
    state_dict = self.get_state(mimi_state)
    layer_state = state_dict["partial"]
    y = self.convtr(x)
    PT = layer_state.shape[-1]
    if PT > 0:
        # Avoid inplace on y if possible, but y is local. 
        # However, layer_state is input.
        # y[..., :PT] += layer_state -> y is modified. layer_state is read. Fine.
        # BUT for safe tracing:
        y_start = y[..., :PT] + layer_state
        y_end = y[..., PT:]
        y = torch.cat([y_start, y_end], dim=-1)

        bias = self.convtr.bias
        for_partial = y[..., -PT:]
        if bias is not None:
            for_partial = for_partial - bias[:, None]
        
        # Out-of-place update
        state_dict["partial"] = for_partial
        y = y[..., :-PT]
    return y

StreamingConv1d.forward = patched_conv1d_forward
StreamingConvTranspose1d.forward = patched_convtr_forward

from onnx_export.wrappers import FlowLMWrapper, MimiWrapper, MimiEncoderWrapper, TextConditionerWrapper
from pocket_tts.modules import conv
import math
def patched_get_extra_padding(x, kernel_size, stride, padding_total=0):
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length
conv.get_extra_padding_for_conv1d = patched_get_extra_padding

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


def export_models(output_dir="onnx_models", language=DEFAULT_LANGUAGE, config=None):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading model...")
    tts_model = TTSModel.load_model(language=language, config=config)
    bundle_name = Path(config).stem if config is not None else language

    tts_model.eval()
    
    # ---------------------------------------------------------
    # Export Mimi Encoder (audio -> latents)
    # ---------------------------------------------------------
    print("Exporting Mimi Encoder...")
    
    mimi_encoder_wrapper = MimiEncoderWrapper(
        tts_model.mimi,
        speaker_proj_weight=tts_model.flow_lm.speaker_proj_weight
    )
    mimi_encoder_wrapper.eval()
    
    # Dummy audio: 1 second at 24kHz
    dummy_audio = torch.randn(1, 1, 24000)
    
    encoder_onnx_path = os.path.join(output_dir, "mimi_encoder.onnx")
    
    torch.onnx.export(
        mimi_encoder_wrapper,
        (dummy_audio,),
        encoder_onnx_path,
        input_names=["audio"],
        output_names=["latents"],
        dynamic_shapes={"audio": {2: "audio_len"}},
        opset_version=18,
        dynamo=True,
        external_data=False,
    )
    print(f"Mimi Encoder exported to {encoder_onnx_path}")
    
    # ---------------------------------------------------------
    # Export Text Conditioner (tokens -> embeddings)
    # ---------------------------------------------------------
    print("Exporting Text Conditioner...")
    
    text_conditioner_wrapper = TextConditionerWrapper(tts_model.flow_lm.conditioner)
    text_conditioner_wrapper.eval()
    
    # Dummy tokens
    dummy_tokens = torch.randint(0, 1000, (1, 20))
    
    conditioner_onnx_path = os.path.join(output_dir, "text_conditioner.onnx")
    
    torch.onnx.export(
        text_conditioner_wrapper,
        (dummy_tokens,),
        conditioner_onnx_path,
        input_names=["token_ids"],
        output_names=["embeddings"],
        dynamic_axes={"token_ids": {1: "seq_len"}},
        opset_version=14,
        dynamo=False,
        external_data=False,
    )
    print(f"Text Conditioner exported to {conditioner_onnx_path}")
    
    # Initialize state with static size sufficient for expected usage
    # 1000 tokens covers ~40s audio or long text prompts
    STATIC_SEQ_LEN = 1000
    
    flow_lm_onnx_path = None
    
    # ---------------------------------------------------------
    # Export Mimi
    # ---------------------------------------------------------
    print("Exporting Mimi...")
    
    mimi_state = init_states(tts_model.mimi, batch_size=1, sequence_length=STATIC_SEQ_LEN)
    mimi_structure = get_state_structure(mimi_state)
    flat_mimi_state = flatten_state(mimi_state)
    write_bundle_metadata(output_dir, tts_model, bundle_name=bundle_name, mimi_state=mimi_state)
    
    
    mimi_wrapper = MimiWrapper(
        tts_model.mimi, 
        mimi_structure,
        emb_std=tts_model.flow_lm.emb_std,
        emb_mean=tts_model.flow_lm.emb_mean
    )
    
    dummy_latent = torch.randn(1, 1, 32)
    mimi_args = (dummy_latent, flat_mimi_state)
    
    mimi_input_names = ["latent"] + [f"state_{i}" for i in range(len(flat_mimi_state))]
    mimi_output_names = ["audio_frame"] + [f"out_state_{i}" for i in range(len(flat_mimi_state))]
    
    # Mimi dynamic axes
    mimi_dynamic_axes = {
        "latent": {1: "seq_len"}
    }
    
    mimi_onnx_path = os.path.join(output_dir, "mimi_decoder.onnx")
    
    torch.onnx.export(
        mimi_wrapper,
        mimi_args,
        mimi_onnx_path,
        input_names=mimi_input_names,
        output_names=mimi_output_names,
        dynamic_axes=mimi_dynamic_axes,
        opset_version=17,
        dynamo=False
    )
    print(f"Mimi exported to {mimi_onnx_path}")
    
    return flow_lm_onnx_path, mimi_onnx_path, tts_model

def verify_export(flow_lm_path, mimi_path, tts_model, output_dir="onnx_models", exact=False):
    print("Verifying export...")
    
    encoder_path = os.path.join(output_dir, "mimi_encoder.onnx")
    conditioner_path = os.path.join(output_dir, "text_conditioner.onnx")
    
    if os.path.exists(encoder_path):
        # ---------------------------------------------------------
        # Verify Mimi Encoder
        # ---------------------------------------------------------
        print("Verifying Mimi Encoder...")
        ort_encoder = ort.InferenceSession(encoder_path)
        
        # Test audio input
        test_audio = torch.randn(1, 1, 24000)  # 1 second
        
        # PyTorch run
        encoder_wrapper = MimiEncoderWrapper(
            tts_model.mimi,
            speaker_proj_weight=tts_model.flow_lm.speaker_proj_weight
        )
        with torch.no_grad():
            pt_encoder_out = encoder_wrapper(test_audio)
        
        # ONNX run
        onnx_encoder_out = ort_encoder.run(None, {"audio": test_audio.numpy()})[0]
        
        compare_outputs("Mimi Encoder output", pt_encoder_out.numpy(), onnx_encoder_out, exact)
    
    if os.path.exists(conditioner_path):
        # ---------------------------------------------------------
        # Verify Text Conditioner
        # ---------------------------------------------------------
        print("Verifying Text Conditioner...")
        ort_conditioner = ort.InferenceSession(conditioner_path)
        
        # Test token input
        test_tokens = torch.randint(0, 1000, (1, 20))
        
        # PyTorch run
        conditioner_wrapper = TextConditionerWrapper(tts_model.flow_lm.conditioner)
        with torch.no_grad():
            pt_conditioner_out = conditioner_wrapper(test_tokens)
        
        # ONNX run
        onnx_conditioner_out = ort_conditioner.run(None, {"token_ids": test_tokens.numpy()})[0]
        
        compare_outputs("Text Conditioner output", pt_conditioner_out.numpy(), onnx_conditioner_out, exact)
    
    if mimi_path and os.path.exists(mimi_path):
        # ---------------------------------------------------------
        # Verify Mimi
        # ---------------------------------------------------------
        ort_session_mimi = ort.InferenceSession(mimi_path)
        
        mimi_state = init_states(tts_model.mimi, batch_size=1, sequence_length=1000)
        flat_mimi_state = flatten_state(mimi_state)
        
        latent = torch.randn(1, 1, tts_model.flow_lm.ldim)
        
        # PyTorch run
        mimi_wrapper = MimiWrapper(
            tts_model.mimi, 
            get_state_structure(mimi_state),
            emb_std=tts_model.flow_lm.emb_std,
            emb_mean=tts_model.flow_lm.emb_mean
        )
        with torch.no_grad():
            pt_mimi_out = mimi_wrapper(latent, flat_mimi_state)
            
        pt_audio = pt_mimi_out[0].numpy()
        pt_mimi_states = [x.numpy() for x in pt_mimi_out[1:]]
        
        # ONNX run
        ort_mimi_inputs = {
            "latent": latent.numpy()
        }
        for i, state_tensor in enumerate(flat_mimi_state):
            ort_mimi_inputs[f"state_{i}"] = state_tensor.numpy()
            
        ort_mimi_outs = ort_session_mimi.run(None, ort_mimi_inputs)
        
        onnx_audio = ort_mimi_outs[0]
        onnx_mimi_states = ort_mimi_outs[1:]
        
        compare_outputs("Mimi audio output", pt_audio, onnx_audio, exact)
        
        for i, (pt_s, onnx_s) in enumerate(zip(pt_mimi_states, onnx_mimi_states)):
            compare_outputs(f"Mimi state {i}", pt_s, onnx_s, exact)
        
        print("Verification successful!")

def main():
    torch.manual_seed(42)
    parser = argparse.ArgumentParser(description="Export Mimi and Conditioner models to ONNX.")
    parser.add_argument("--output_dir", "-o", type=str, default="onnx_models", help="Directory for output ONNX files")
    parser.add_argument("--language", type=str, default=DEFAULT_LANGUAGE, help="Model language/config name.")
    parser.add_argument("--config", type=str, default=None, help="Path to a local YAML config file.")
    parser.add_argument("--exact", action="store_true", help="Require exact torch vs ONNX equality.")
    args = parser.parse_args()
    
    flow, mimi, model = export_models(output_dir=args.output_dir, language=args.language, config=args.config)
    verify_export(flow, mimi, model, output_dir=args.output_dir, exact=args.exact)

if __name__ == "__main__":
    main()
