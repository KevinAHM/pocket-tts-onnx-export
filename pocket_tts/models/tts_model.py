import copy
import logging
import math
import os
import queue
import statistics
import threading
import time
from functools import lru_cache
from pathlib import Path

import safetensors
import safetensors.torch
import scipy.io.wavfile
import torch
from torch import nn
from torch.nn import functional as F
from typing_extensions import Self

from pocket_tts.conditioners.base import TokenizedText
from pocket_tts.data.audio import audio_read
from pocket_tts.data.audio_utils import convert_audio
from pocket_tts.default_parameters import (
    DEFAULT_EOS_THRESHOLD,
    DEFAULT_LANGUAGE,
    DEFAULT_LSD_DECODE_STEPS,
    DEFAULT_NOISE_CLAMP,
    DEFAULT_TEMPERATURE,
    MAX_TOKEN_PER_CHUNK,
)
from pocket_tts.models.flow_lm import FlowLMModel
from pocket_tts.models.mimi import MimiModel
from pocket_tts.modules import mimi_transformer
from pocket_tts.modules.dummy_quantizer import DummyQuantizer
from pocket_tts.modules.seanet import SEANetDecoder, SEANetEncoder
from pocket_tts.modules.stateful_module import StatefulModule, increment_steps, init_states
from pocket_tts.quantization import RECOMMENDED_CONFIG, apply_dynamic_int8
from pocket_tts.utils.config import CONFIGS_DIR, Config, load_config
from pocket_tts.utils.utils import (
    PREDEFINED_VOICES,
    _ORIGINS_OF_PREDEFINED_VOICES,
    DEBUG_MIMI,
    display_execution_time,
    download_if_necessary,
    get_predefined_voice,
    load_predefined_voice,
    size_of_dict,
)
from pocket_tts.utils.weights_loading import get_flow_lm_state_dict, get_mimi_state_dict

torch.set_num_threads(1)
logger = logging.getLogger(__name__)

VOICE_CLONING_REQUIRED = (
    "The gated voice-cloning weights from https://huggingface.co/kyutai/pocket-tts "
    "must be available locally. Accept the model terms and authenticate with "
    "`uvx hf auth login`, then retry."
)


def _normalize_language_name(language: str | None) -> str | None:
    if language is None:
        return None
    if language == "english":
        return DEFAULT_LANGUAGE
    return language.replace("_2026_", "_2026-")


class TTSModel(nn.Module):
    _TOKENS_PER_SECOND_ESTIMATE = 3.0
    _GEN_SECONDS_PADDING = 2.0

    def __init__(
        self,
        flow_lm: FlowLMModel,
        temp: float,
        lsd_decode_steps: int,
        noise_clamp: float | None,
        eos_threshold,
        config: Config,
        origin: Path | None = None,
        pad_with_spaces_for_short_inputs: bool = False,
        model_recommended_frames_after_eos: int | None = None,
        remove_semicolons: bool = False,
    ):
        super().__init__()
        self.flow_lm = flow_lm
        self.temp = temp
        self.lsd_decode_steps = lsd_decode_steps
        self.noise_clamp = noise_clamp
        self.eos_threshold = eos_threshold
        self.config = config
        self.origin = origin
        self.pad_with_spaces_for_short_inputs = pad_with_spaces_for_short_inputs
        self.model_recommended_frames_after_eos = model_recommended_frames_after_eos
        self.remove_semicolons = remove_semicolons

    @property
    def device(self) -> str:
        return next(self.parameters()).device.type

    @property
    def sample_rate(self) -> int:
        return self.config.mimi.sample_rate

    @classmethod
    def _from_pydantic_config(
        cls,
        config: Config,
        temp,
        lsd_decode_steps,
        noise_clamp: float | None,
        eos_threshold,
        origin: Path | None,
    ) -> Self:
        flow_lm = FlowLMModel.from_pydantic_config(
            config.flow_lm,
            latent_dim=config.mimi.quantizer.dimension,
            insert_bos_before_voice=config.flow_lm.insert_bos_before_voice,
        )
        return cls(
            flow_lm,
            temp,
            lsd_decode_steps,
            noise_clamp,
            eos_threshold,
            config,
            origin=origin,
            pad_with_spaces_for_short_inputs=config.pad_with_spaces_for_short_inputs,
            model_recommended_frames_after_eos=config.model_recommended_frames_after_eos,
            remove_semicolons=config.remove_semicolons,
        )

    @classmethod
    def _from_pydantic_config_with_weights(
        cls,
        config: Config,
        temp,
        lsd_decode_steps,
        noise_clamp: float | None,
        eos_threshold,
        origin: Path | None = None,
    ) -> Self:
        tts_model = cls._from_pydantic_config(
            config, temp, lsd_decode_steps, noise_clamp, eos_threshold, origin=origin
        )
        tts_model.flow_lm.speaker_proj_weight = torch.nn.Parameter(
            torch.zeros(
                (
                    config.flow_lm.transformer.d_model,
                    config.mimi.inner_dim or config.mimi.seanet.dimension,
                ),
                dtype=torch.float32,
            )
        )
        if config.flow_lm.weights_path is not None:
            if config.mimi.weights_path is None:
                raise ValueError(
                    "If you specify flow_lm.weights_path you should specify mimi.weights_path"
                )
            logger.info("Loading FlowLM weights from %s", config.flow_lm.weights_path)
            state_dict_flowlm = get_flow_lm_state_dict(
                download_if_necessary(config.flow_lm.weights_path)
            )
            tts_model.flow_lm.load_state_dict(state_dict_flowlm, strict=True)

        mimi_config = config.mimi.model_dump()
        encoder = SEANetEncoder(**mimi_config["seanet"])
        decoder = SEANetDecoder(**mimi_config["seanet"])
        encoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
        decoder_transformer = mimi_transformer.ProjectedTransformer(**mimi_config["transformer"])
        quantizer = DummyQuantizer(**mimi_config["quantizer"])

        tts_model.mimi = MimiModel(
            encoder,
            decoder,
            quantizer,
            channels=mimi_config["channels"],
            sample_rate=mimi_config["sample_rate"],
            frame_rate=mimi_config["frame_rate"],
            encoder_frame_rate=mimi_config["sample_rate"] / encoder.hop_length,
            inner_dim=mimi_config["inner_dim"],
            outer_dim=mimi_config["outer_dim"],
            encoder_transformer=encoder_transformer,
            decoder_transformer=decoder_transformer,
        ).to(device="cpu")

        if config.mimi.weights_path is not None:
            if config.flow_lm.weights_path is None:
                raise ValueError(
                    "If you specify mimi.weights_path you should specify flow_lm.weights_path"
                )
            logger.info("Loading Mimi weights from %s", config.mimi.weights_path)
            mimi_state = get_mimi_state_dict(download_if_necessary(config.mimi.weights_path))
            tts_model.mimi.load_state_dict(mimi_state, strict=True)

        tts_model.mimi.eval()

        if config.weights_path is not None:
            logger.info("Loading TTSModel weights from %s", config.weights_path)
            try:
                weights_file = download_if_necessary(config.weights_path)
            except Exception as exc:
                raise RuntimeError(VOICE_CLONING_REQUIRED) from exc

            state_dict = safetensors.torch.load_file(weights_file)
            tts_model.load_state_dict(state_dict, strict=True)

        if config.flow_lm.weights_path is None and config.weights_path is None:
            logger.warning(
                "No weights_path specified for FlowLM or TTSModel, model is uninitialized!"
            )

        size_in_mb = size_of_dict(tts_model.state_dict()) // 1e6
        if os.environ.get("POCKET_TTS_SAVE_WEIGHTS", "0") == "1":
            save_path = "./model.safetensors"
            safetensors.torch.save_file(tts_model.state_dict(), save_path)
            logger.info("Saved TTSModel weights to %s", save_path)
        logging.info("TTS Model loaded successfully. Its size is %d MB", size_in_mb)

        for top_module in (tts_model.flow_lm, tts_model.mimi):
            for module_name, module in top_module.named_modules():
                if not isinstance(module, StatefulModule):
                    continue
                module._module_absolute_name = module_name

        return tts_model

    @classmethod
    def load_model(
        cls,
        language: str | None = None,
        config: str | Path | None = None,
        temp: float | int = DEFAULT_TEMPERATURE,
        lsd_decode_steps: int = DEFAULT_LSD_DECODE_STEPS,
        noise_clamp: float | int | None = DEFAULT_NOISE_CLAMP,
        eos_threshold: float = DEFAULT_EOS_THRESHOLD,
        quantize: bool = False,
    ) -> Self:
        if config is not None and language is not None:
            raise ValueError("Cannot specify both config and language.")

        language = _normalize_language_name(language)
        if config is None and language is None:
            language = DEFAULT_LANGUAGE

        if language is not None:
            config_path = CONFIGS_DIR / f"{language}.yaml"
        else:
            config_path = Path(config)
            if config_path.suffix not in (".yaml", ".yml"):
                config_path = CONFIGS_DIR / f"{config_path}.yaml"

        loaded_config = load_config(config_path)
        logger.info("Loading model from config at %s...", config_path)

        tts_model = cls._from_pydantic_config_with_weights(
            loaded_config,
            temp,
            lsd_decode_steps,
            noise_clamp,
            eos_threshold,
            origin=config_path,
        )

        if quantize:
            apply_dynamic_int8(tts_model.flow_lm, RECOMMENDED_CONFIG)

        return tts_model

    def _run_flow_lm_and_increment_step(
        self,
        model_state: dict,
        text_tokens: torch.Tensor | None = None,
        backbone_input_latents: torch.Tensor | None = None,
        audio_conditioning: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if text_tokens is None:
            text_tokens = torch.zeros((1, 0), dtype=torch.int64, device=self.flow_lm.device)
        if backbone_input_latents is None:
            backbone_input_latents = torch.empty(
                (1, 0, self.flow_lm.ldim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )
        if audio_conditioning is None:
            audio_conditioning = torch.empty(
                (1, 0, self.flow_lm.dim), dtype=self.flow_lm.dtype, device=self.flow_lm.device
            )

        output = self._run_flow_lm(
            text_tokens=text_tokens,
            backbone_input_latents=backbone_input_latents,
            model_state=model_state,
            audio_conditioning=audio_conditioning,
        )
        increment_by = (
            text_tokens.shape[1] + backbone_input_latents.shape[1] + audio_conditioning.shape[1]
        )
        increment_steps(self.flow_lm, model_state, increment=increment_by)
        return output

    def _run_flow_lm(
        self,
        model_state: dict,
        text_tokens: torch.Tensor,
        backbone_input_latents: torch.Tensor,
        audio_conditioning: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_embeddings = self.flow_lm.conditioner(TokenizedText(text_tokens))
        text_embeddings = torch.cat([text_embeddings, audio_conditioning], dim=1)

        output_embeddings, is_eos = self.flow_lm._sample_next_latent(
            backbone_input_latents,
            text_embeddings,
            model_state=model_state,
            lsd_decode_steps=self.lsd_decode_steps,
            temp=self.temp,
            noise_clamp=self.noise_clamp,
            eos_threshold=self.eos_threshold,
        )
        return output_embeddings[:, None, :], is_eos

    def _decode_and_dump(self, encoded: torch.Tensor, filename: str):
        mimi_state = init_states(self.mimi, batch_size=1, sequence_length=10000)
        latent_to_decode = encoded if encoded.shape[1] != self.mimi.quantizer.dimension else self.mimi.quantizer(encoded)
        restored_audio = self.mimi.decode_from_latent(latent_to_decode, mimi_state)
        scipy.io.wavfile.write(filename, self.sample_rate, restored_audio.numpy())
        logger.info("Saved restored audio from Mimi encoding to %s for debugging", filename)

    def _encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        encoded = self.mimi.encode_to_latent(audio)
        if DEBUG_MIMI:
            self._decode_and_dump(encoded, "debug_encoded_latent_decoded.wav")
        latents = encoded.transpose(-1, -2).to(torch.float32)
        return F.linear(latents, self.flow_lm.speaker_proj_weight)

    def _expand_kv_cache(self, model_state: dict, sequence_length: int) -> None:
        for module_state in model_state.values():
            if "cache" not in module_state:
                continue
            cache = module_state["cache"]
            current_length = cache.shape[2]
            if current_length >= sequence_length:
                continue
            expanded_cache = torch.full(
                (
                    cache.shape[0],
                    cache.shape[1],
                    sequence_length,
                    cache.shape[3],
                    cache.shape[4],
                ),
                float("NaN"),
                device=cache.device,
                dtype=cache.dtype,
            )
            expanded_cache[:, :, :current_length, :, :] = cache
            module_state["cache"] = expanded_cache

    def _flow_lm_current_end(self, model_state: dict) -> int:
        for module_state in model_state.values():
            if "current_end" in module_state:
                return int(module_state["current_end"].shape[0])
            if "offset" in module_state and "end_offset" not in module_state:
                return int(module_state["offset"].view(-1)[0].item())
        raise ValueError("Could not infer the FlowLM attention offset from model state.")

    @torch.no_grad
    def _decode_audio_worker(
        self,
        latents_queue: queue.Queue,
        result_queue: queue.Queue,
        mimi_sequence_length: int,
        mimi_steps_per_latent: int,
    ):
        try:
            mimi_state = init_states(self.mimi, batch_size=1, sequence_length=mimi_sequence_length)
            while True:
                latent = latents_queue.get()
                if latent is None:
                    break
                mimi_decoding_input = latent * self.flow_lm.emb_std + self.flow_lm.emb_mean
                transposed = mimi_decoding_input.transpose(-1, -2)
                quantized = self.mimi.quantizer(transposed)

                t = time.monotonic()
                audio_frame = self.mimi.decode_from_latent(quantized, mimi_state)
                increment_steps(self.mimi, mimi_state, increment=mimi_steps_per_latent)
                audio_frame_duration = audio_frame.shape[2] / self.config.mimi.sample_rate
                logger.debug(
                    " " * 30 + "Decoded %d ms of audio with mimi in %d ms",
                    int(audio_frame_duration * 1000),
                    int((time.monotonic() - t) * 1000),
                )
                result_queue.put(("chunk", audio_frame))
                latents_queue.task_done()

            result_queue.put(("done", None))
        except Exception as exc:
            result_queue.put(("error", exc))

    @torch.no_grad
    def generate_audio(
        self,
        model_state: dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ) -> torch.Tensor:
        audio_chunks = []
        for chunk in self.generate_audio_stream(
            model_state=model_state,
            text_to_generate=text_to_generate,
            max_tokens=max_tokens,
            frames_after_eos=frames_after_eos,
            copy_state=copy_state,
        ):
            audio_chunks.append(chunk)
        return torch.cat(audio_chunks, dim=0)

    @torch.no_grad
    def generate_audio_stream(
        self,
        model_state: dict,
        text_to_generate: str,
        max_tokens: int = MAX_TOKEN_PER_CHUNK,
        frames_after_eos: int | None = None,
        copy_state: bool = True,
    ):
        if frames_after_eos is None:
            frames_after_eos = self.model_recommended_frames_after_eos

        chunks = split_into_best_sentences(
            self.flow_lm.conditioner.tokenizer,
            text_to_generate,
            max_tokens,
            self.pad_with_spaces_for_short_inputs,
            remove_semicolons=self.remove_semicolons,
        )

        for chunk in chunks:
            _, frames_after_eos_guess = prepare_text_prompt(
                chunk, self.pad_with_spaces_for_short_inputs, self.remove_semicolons
            )
            frames_after_eos_guess += 2
            effective_frames = (
                frames_after_eos if frames_after_eos is not None else frames_after_eos_guess
            )
            yield from self._generate_audio_stream_short_text(
                model_state=model_state,
                text_to_generate=chunk,
                frames_after_eos=effective_frames,
                copy_state=copy_state,
            )

    @torch.no_grad
    def _generate_audio_stream_short_text(
        self, model_state: dict, text_to_generate: str, frames_after_eos: int, copy_state: bool
    ):
        if copy_state:
            model_state = copy.deepcopy(model_state)

        prepared = self.flow_lm.conditioner.prepare(text_to_generate)
        token_count = prepared.tokens.shape[1]
        max_gen_len = self._estimate_max_gen_len(token_count)
        mimi_steps_per_latent = int(self.mimi.encoder_frame_rate / self.mimi.frame_rate)
        mimi_sequence_length = max_gen_len * mimi_steps_per_latent

        latents_queue = queue.Queue()
        result_queue = queue.Queue()

        decoder_thread = threading.Thread(
            target=self._decode_audio_worker,
            args=(latents_queue, result_queue, mimi_sequence_length, mimi_steps_per_latent),
            daemon=True,
        )
        t_generating = time.monotonic()
        decoder_thread.start()

        self._generate(
            model_state=model_state,
            prepared=prepared,
            max_gen_len=max_gen_len,
            frames_after_eos=frames_after_eos,
            latents_queue=latents_queue,
            result_queue=result_queue,
        )

        total_generated_samples = 0
        while True:
            result = result_queue.get()
            if result[0] == "chunk":
                audio_chunk = result[1]
                total_generated_samples += audio_chunk.shape[-1]
                yield audio_chunk[0, 0]
            elif result[0] == "done":
                break
            elif result[0] == "error":
                with display_execution_time("Waiting for mimi decoder to finish"):
                    decoder_thread.join()
                raise result[1]

        with display_execution_time("Waiting for mimi decoder to finish"):
            decoder_thread.join()

        duration_generated_audio = int(
            total_generated_samples * 1000 / self.config.mimi.sample_rate
        )
        generation_time = int((time.monotonic() - t_generating) * 1000)
        real_time_factor = duration_generated_audio / generation_time

        logger.info(
            "Generated: %d ms of audio in %d ms so %.2fx faster than real-time",
            duration_generated_audio,
            generation_time,
            real_time_factor,
        )

    @torch.no_grad
    def _generate(
        self,
        model_state: dict,
        prepared: TokenizedText,
        max_gen_len: int,
        frames_after_eos: int,
        latents_queue: queue.Queue,
        result_queue: queue.Queue,
    ):
        token_count = prepared.tokens.shape[1]
        current_end = self._flow_lm_current_end(model_state)
        required_len = current_end + token_count + max_gen_len
        self._expand_kv_cache(model_state, sequence_length=required_len)

        with display_execution_time("Prompting text"):
            self._run_flow_lm_and_increment_step(
                model_state=model_state, text_tokens=prepared.tokens
            )

        def run_generation():
            try:
                self._autoregressive_generation(
                    model_state, max_gen_len, frames_after_eos, latents_queue
                )
            except Exception as exc:
                logger.error("Error in autoregressive generation: %s", exc)
                if latents_queue is not None:
                    latents_queue.put(None)
                if result_queue is not None:
                    result_queue.put(("error", exc))

        generation_thread = threading.Thread(target=run_generation, daemon=True)
        generation_thread.start()

    @torch.no_grad
    def _autoregressive_generation(
        self, model_state: dict, max_gen_len: int, frames_after_eos: int, latents_queue: queue.Queue
    ):
        backbone_input = torch.full(
            (1, 1, self.flow_lm.ldim),
            fill_value=float("NaN"),
            device=next(iter(self.flow_lm.parameters())).device,
            dtype=self.flow_lm.dtype,
        )
        steps_times = []
        eos_step = None
        for generation_step in range(max_gen_len):
            with display_execution_time("Generating latent", print_output=False) as timer:
                next_latent, is_eos = self._run_flow_lm_and_increment_step(
                    model_state=model_state, backbone_input_latents=backbone_input
                )
                if is_eos.item() and eos_step is None:
                    eos_step = generation_step
                if eos_step is not None and generation_step >= eos_step + frames_after_eos:
                    break

                latents_queue.put(next_latent)
                backbone_input = next_latent
            steps_times.append(timer.elapsed_time_ms)
        else:
            if os.environ.get("KPOCKET_TTS_ERROR_WITHOUT_EOS", "0") == "1":
                raise RuntimeError("Generation reached maximum length without EOS!")
            logger.warning(
                "Maximum generation length reached without EOS, this very often indicates an error."
            )

        latents_queue.put(None)
        logger.info("Average generation step time: %d ms", int(statistics.mean(steps_times)))

    @lru_cache(maxsize=2)
    def _cached_get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict:
        return self.get_state_for_audio_prompt(audio_conditioning, truncate)

    @torch.no_grad
    def get_state_for_audio_prompt(
        self, audio_conditioning: Path | str | torch.Tensor, truncate: bool = False
    ) -> dict:
        if isinstance(audio_conditioning, (str, Path)) and str(audio_conditioning).endswith(
            ".safetensors"
        ):
            if isinstance(audio_conditioning, str):
                audio_conditioning = download_if_necessary(audio_conditioning)
            return _import_model_state(audio_conditioning)

        if isinstance(audio_conditioning, str) and audio_conditioning in _ORIGINS_OF_PREDEFINED_VOICES:
            if self.origin is not None and self.origin.is_relative_to(CONFIGS_DIR):
                return _import_model_state(
                    download_if_necessary(
                        get_predefined_voice(language=self.origin.stem, name=audio_conditioning)
                    )
                )
            prompt = load_predefined_voice(audio_conditioning)
        else:
            if isinstance(audio_conditioning, str):
                audio_conditioning = download_if_necessary(audio_conditioning)

            if isinstance(audio_conditioning, Path):
                audio, conditioning_sample_rate = audio_read(audio_conditioning)
                if truncate:
                    max_samples = int(30 * conditioning_sample_rate)
                    if audio.shape[-1] > max_samples:
                        audio = audio[..., :max_samples]
                        logger.info("Audio truncated to first 30 seconds (%d samples)", max_samples)

                audio_conditioning = convert_audio(
                    audio, conditioning_sample_rate, self.config.mimi.sample_rate, 1
                )

            with display_execution_time("Encoding audio prompt"):
                prompt = self._encode_audio(audio_conditioning.unsqueeze(0).to(self.device))

        if self.flow_lm.insert_bos_before_voice:
            prompt = torch.cat([self.flow_lm.bos_before_voice, prompt], dim=1)

        model_state = init_states(self.flow_lm, batch_size=1, sequence_length=prompt.shape[1])

        with display_execution_time("Prompting audio"):
            self._run_flow_lm_and_increment_step(model_state=model_state, audio_conditioning=prompt)

        logger.info(
            "Size of the model state for audio prompt: %d MB", size_of_dict(model_state) // 1e6
        )
        return model_state

    def _estimate_max_gen_len(self, token_count: int) -> int:
        gen_len_sec = token_count / self._TOKENS_PER_SECOND_ESTIMATE + self._GEN_SECONDS_PADDING
        return math.ceil(gen_len_sec * self.config.mimi.frame_rate)


def prepare_text_prompt(
    text: str, pad_with_spaces_for_short_inputs: bool, remove_semicolons: bool
) -> tuple[str, int]:
    text = text.strip()
    if text == "":
        raise ValueError("Text prompt cannot be empty")
    text = text.replace("\n", " ").replace("\r", " ").replace("  ", " ")
    if remove_semicolons:
        text = text.replace(";", ",")
    number_of_words = len(text.split())
    frames_after_eos_guess = 3 if number_of_words <= 4 else 1

    if not text[0].isupper():
        text = text[0].upper() + text[1:]

    if text[-1].isalnum():
        text = text + "."

    if pad_with_spaces_for_short_inputs and len(text.split()) < 5:
        text = " " * 8 + text

    return text, frames_after_eos_guess


def _find_boundary_indices(list_of_tokens: list[int], boundary_tokens: list[int]) -> list[int]:
    indices = [0]
    previous_was_boundary = False
    for idx, token in enumerate(list_of_tokens):
        if token in boundary_tokens:
            previous_was_boundary = True
        else:
            if previous_was_boundary:
                indices.append(idx)
            previous_was_boundary = False
    indices.append(len(list_of_tokens))
    return indices


def _segments_from_boundaries(
    list_of_tokens: list[int], boundary_indices: list[int], tokenizer
) -> list[tuple[int, str]]:
    segments = []
    for i in range(len(boundary_indices) - 1):
        start = boundary_indices[i]
        end = boundary_indices[i + 1]
        text = tokenizer.sp.decode(list_of_tokens[start:end])
        segments.append((end - start, text))
    return segments


def split_into_best_sentences(
    tokenizer,
    text_to_generate: str,
    max_tokens: int,
    pad_with_spaces_for_short_inputs: bool,
    remove_semicolons: bool,
) -> list[str]:
    text_to_generate, _ = prepare_text_prompt(
        text_to_generate, pad_with_spaces_for_short_inputs, remove_semicolons
    )
    text_to_generate = text_to_generate.strip()
    tokens = tokenizer(text_to_generate)
    list_of_tokens = tokens.tokens[0].tolist()

    _, *end_of_sentence_tokens = tokenizer(".!...?").tokens[0].tolist()
    sentence_boundaries = _find_boundary_indices(list_of_tokens, end_of_sentence_tokens)
    nb_tokens_and_sentences = _segments_from_boundaries(
        list_of_tokens, sentence_boundaries, tokenizer
    )

    _, *fallback_tokens = tokenizer(",;:").tokens[0].tolist()
    refined_segments = []
    for nb_tokens, text in nb_tokens_and_sentences:
        if nb_tokens <= max_tokens:
            refined_segments.append((nb_tokens, text))
            continue

        sub_tokens = tokenizer(text.strip()).tokens[0].tolist()
        sub_boundaries = _find_boundary_indices(sub_tokens, fallback_tokens)
        sub_segments = _segments_from_boundaries(sub_tokens, sub_boundaries, tokenizer)
        if len(sub_segments) > 1:
            refined_segments.extend(sub_segments)
        else:
            refined_segments.append((nb_tokens, text))

    chunks = []
    current_chunk = ""
    current_nb_of_tokens_in_chunk = 0
    for nb_tokens, sentence in refined_segments:
        if current_chunk == "":
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
            continue

        if current_nb_of_tokens_in_chunk + nb_tokens > max_tokens:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
            current_nb_of_tokens_in_chunk = nb_tokens
        else:
            current_chunk += " " + sentence
            current_nb_of_tokens_in_chunk += nb_tokens

    if current_chunk != "":
        chunks.append(current_chunk.strip())

    for chunk in chunks:
        chunk_tokens = tokenizer(chunk.strip()).tokens[0].tolist()
        if len(chunk_tokens) > max_tokens:
            logger.warning(
                "Chunk has %d tokens (max %d), generation may skip words: '%.50s...'",
                len(chunk_tokens),
                max_tokens,
                chunk,
            )

    return chunks


def export_model_state(model_state: dict[str, dict[str, torch.Tensor]], dest: str | Path):
    dict_to_store = {}
    for module_name, module_state in model_state.items():
        for key, tensor_value in module_state.items():
            dict_to_store[f"{module_name}/{key}"] = tensor_value
    safetensors.torch.save_file(dict_to_store, dest)


def _import_model_state(source: str | Path) -> dict[str, dict[str, torch.Tensor]]:
    result = {}
    with safetensors.safe_open(source, framework="pt") as handle:
        for key in handle.keys():
            module_name, tensor_key = key.split("/", 1)
            result.setdefault(module_name, {})
            result[module_name][tensor_key] = handle.get_tensor(key)

    for module_state in result.values():
        if "offset" in module_state and "end_offset" not in module_state and "current_end" not in module_state:
            offset = module_state.pop("offset")
            step_count = int(offset.view(-1)[0].item())
            module_state["current_end"] = torch.zeros((step_count,), dtype=offset.dtype, device=offset.device)

    return result
