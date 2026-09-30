"""Speech foundation-model wrappers: WavLM, Wav2Vec2, HuBERT.

HuggingFace returns ``L+1`` hidden states. Index 0 is the CNN / feature-encoder
output. The paper indexes Transformer layers ``l = 1..L``, so every public
method that feeds the hierarchical losses or the downstream fusion drops layer 0.
"""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


FAMILY_TO_HF_TYPE = {
    "wavlm": "wavlm",
    "wav2vec2": "wav2vec2",
    "hubert": "hubert",
}


def _import_model_class(family: str):
    family = family.lower()
    if family == "wavlm":
        from transformers import WavLMModel
        return WavLMModel
    if family == "wav2vec2":
        from transformers import Wav2Vec2Model
        return Wav2Vec2Model
    if family == "hubert":
        from transformers import HubertModel
        return HubertModel
    raise ValueError(f"Unknown backbone family {family!r}. Supported: wavlm, wav2vec2, hubert")


class SpeechFoundationBackbone(nn.Module):
    """A single wrapper around a HuggingFace speech SSL encoder."""

    def __init__(
        self,
        family: str,
        pretrained_name: str,
        *,
        local_path: str | None = None,
        checkpoint: str | None = None,
        freeze: bool = True,
        padding_mode: str = "reference",
    ):
        super().__init__()
        self.family = family.lower()
        if padding_mode not in {"reference", "isolated"}:
            raise ValueError(f"Unknown padding_mode {padding_mode!r}")
        self.padding_mode = padding_mode
        if self.family not in FAMILY_TO_HF_TYPE:
            raise ValueError(f"Unknown backbone family {family!r}")
        source = _resolve_load_source(local_path, checkpoint, pretrained_name)
        model_cls = _import_model_class(self.family)
        self.model = _load_or_cache_pretrained(model_cls, pretrained_name, source)
        actual_type = str(getattr(self.model.config, "model_type", "")).lower()
        expected_type = FAMILY_TO_HF_TYPE[self.family]
        if actual_type != expected_type:
            raise ValueError(
                f"Backbone family mismatch: requested {expected_type}, loaded {actual_type} from {source}"
            )
        self.model.config.output_hidden_states = True
        self._hidden_size = int(self.model.config.hidden_size)
        self._num_transformer_layers = int(self.model.config.num_hidden_layers)
        # Group-norm feature extractors (plain facebook/wav2vec2-large) were
        # pretrained on unpadded batches; HuggingFace advises against passing a
        # mask. Layer-norm families (WavLM Large, HuBERT Large, wav2vec2-lv60)
        # expect one. Passing the wrong choice changes every downstream number.
        self.feat_extract_norm = str(getattr(self.model.config, "feat_extract_norm", "layer"))
        self.use_attention_mask = self.feat_extract_norm != "group"
        self._frozen = False
        self._attack_mode = False
        disable_representation_stochasticity(self)
        print(
            f"  {self.family}: L={self._num_transformer_layers} D={self._hidden_size} "
            f"feat_extract_norm={self.feat_extract_norm} "
            f"attention_mask={self.use_attention_mask}"
        )
        if freeze:
            self.freeze()

    @property
    def hidden_size(self) -> int:
        return self._hidden_size

    @property
    def num_layers(self) -> int:
        """Number of Transformer layers ``L`` (excludes the CNN extractor)."""
        return self._num_transformer_layers

    def freeze(self) -> "SpeechFoundationBackbone":
        for p in self.parameters():
            p.requires_grad = False
        self.eval()
        self._frozen = True
        return self

    def unfreeze(self) -> "SpeechFoundationBackbone":
        for p in self.parameters():
            p.requires_grad = True
        self._frozen = False
        return self

    def enable_attack_mode(self) -> None:
        """Build a graph through frozen weights so PGD can differentiate the waveform."""
        self._attack_mode = True

    def disable_attack_mode(self) -> None:
        self._attack_mode = False

    @property
    def grad_context(self):
        if self._attack_mode:
            return torch.enable_grad()
        if self._frozen:
            return torch.no_grad()
        return nullcontext()

    def train(self, mode: bool = True):
        if self._frozen:
            return super().train(False)
        return super().train(mode)

    def enable_gradient_checkpointing(self) -> None:
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable()

    def attention_mask(self, waveforms: torch.Tensor, lengths: torch.Tensor | None) -> torch.Tensor | None:
        if lengths is None or not self.use_attention_mask:
            return None
        time_steps = waveforms.size(1)
        arange = torch.arange(time_steps, device=waveforms.device)[None, :]
        return (arange < lengths[:, None]).long()

    def feature_lengths(self, lengths: torch.Tensor) -> torch.Tensor:
        return self.model._get_feat_extract_output_lengths(lengths)

    def forward(
        self, waveforms: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """All hidden states including h0. ``hidden_states: [L+1, B, T, D]``."""
        attention_mask = self.attention_mask(waveforms, lengths)
        feat_lengths = None if lengths is None else self.feature_lengths(lengths)
        with self.grad_context:
            # Group-norm Wav2Vec2 does not support an attention mask. Running a
            # zero-padded batch together lets padding affect both group norm and
            # self-attention, including valid frames. Evaluate each true waveform
            # separately, then pad only the resulting hidden states.
            if (
                self.padding_mode == "isolated"
                and not self.use_attention_mask
                and lengths is not None
                and bool((lengths != waveforms.size(1)).any())
            ):
                outputs_by_sample = []
                max_frames = int(feat_lengths.max().item())
                for i in range(waveforms.size(0)):
                    n = int(lengths[i].item())
                    one = self.model(
                        waveforms[i:i + 1, :n], output_hidden_states=True
                    )
                    states = torch.stack(one.hidden_states, dim=0)
                    outputs_by_sample.append(
                        F.pad(states, (0, 0, 0, max_frames - states.size(2)))
                    )
                hidden_states = torch.cat(outputs_by_sample, dim=1)
            else:
                outputs = self.model(
                    waveforms,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                )
                hidden_states = torch.stack(outputs.hidden_states, dim=0)
        return hidden_states, feat_lengths

    def transformer_states(
        self, waveforms: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Transformer layers ``l = 1..L`` and the corresponding frame mask.

        Returns ``states [L, B, T, D]`` and boolean ``frame_mask [B, T]``.
        """
        states, feat_lengths = self.forward(waveforms, lengths)
        states = states[1:]
        if states.size(0) != self.num_layers:
            raise RuntimeError(
                f"Expected {self.num_layers} Transformer outputs after excluding h0, "
                f"received {states.size(0)}"
            )
        frame_mask = (
            torch.arange(states.size(2), device=states.device).unsqueeze(0)
            < feat_lengths.unsqueeze(1)
        )
        return states, frame_mask

    def save_pretrained(self, path: str | Path) -> Path:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        return path


def transformer_states(encoder, waveforms, lengths):
    """Module-level alias used by the Stage 1 attack and trainer."""
    return encoder.transformer_states(waveforms, lengths)


def disable_representation_stochasticity(encoder) -> None:
    model = encoder.model if isinstance(encoder, SpeechFoundationBackbone) else encoder
    root = encoder if isinstance(encoder, nn.Module) else model
    for module in root.modules():
        if isinstance(module, nn.Dropout):
            module.p = 0.0
        probability = getattr(module, "dropout", None)
        if isinstance(probability, (float, int)):
            module.dropout = 0.0
        layerdrop = getattr(module, "layerdrop", None)
        if isinstance(layerdrop, (float, int)):
            module.layerdrop = 0.0
    config = model.config if hasattr(model, "config") else getattr(root, "config", None)
    if config is None:
        return
    for name in (
        "hidden_dropout", "attention_dropout", "activation_dropout",
        "feat_proj_dropout", "final_dropout", "layerdrop",
        "mask_time_prob", "mask_feature_prob",
    ):
        if hasattr(config, name):
            setattr(config, name, 0.0)
    if hasattr(config, "apply_spec_augment"):
        config.apply_spec_augment = False


def _resolve_load_source(local_path, checkpoint, pretrained_name) -> str:
    if checkpoint:
        if not Path(str(checkpoint)).exists():
            raise FileNotFoundError(f"backbone checkpoint not found: {checkpoint}")
        return str(checkpoint)
    if local_path:
        if not Path(str(local_path)).is_dir():
            raise FileNotFoundError(f"local pretrained model directory not found: {local_path}")
        return str(local_path)
    return pretrained_name


def _load_or_cache_pretrained(model_cls, model_name: str, source: str):
    source_path = Path(source)
    if source_path.is_dir():
        print(f"  Loading backbone from {source_path}")
        return model_cls.from_pretrained(str(source_path))
    print(f"  Loading backbone {source}")
    return model_cls.from_pretrained(source)


def build_backbone(
    family: str,
    pretrained_name: str,
    *,
    local_path: str | None = None,
    checkpoint: str | None = None,
    freeze: bool = True,
    padding_mode: str = "reference",
) -> SpeechFoundationBackbone:
    """Build ``theta_b`` (no checkpoint) or ``theta_r`` (checkpoint directory)."""
    if checkpoint:
        ckpt = Path(checkpoint)
        if not ckpt.exists():
            raise FileNotFoundError(
                f"backbone checkpoint not found: {ckpt}. "
                "Stage 1 must finish and write theta_r (or best/) first."
            )
        if ckpt.is_dir() and (
            (ckpt / "config.json").is_file()
            or (ckpt / "model.safetensors").is_file()
            or (ckpt / "pytorch_model.bin").is_file()
        ):
            return SpeechFoundationBackbone(
                family, pretrained_name, local_path=str(ckpt), freeze=freeze,
                padding_mode=padding_mode,
            )
        backbone = SpeechFoundationBackbone(
            family, pretrained_name, local_path=local_path, freeze=False,
            padding_mode=padding_mode,
        )
        _load_state_file(backbone.model, ckpt)
        if freeze:
            backbone.freeze()
        else:
            disable_representation_stochasticity(backbone)
        return backbone
    return SpeechFoundationBackbone(
        family, pretrained_name, local_path=local_path, freeze=freeze,
        padding_mode=padding_mode,
    )


def _load_state_file(model: nn.Module, path: Path) -> None:
    path = Path(path)
    if path.is_dir():
        candidates = (
            sorted(path.glob("*.safetensors"))
            + [p for p in sorted(path.glob("*.pt")) if p.name != "training_state.pt"]
            + sorted(path.glob("*.bin"))
        )
        if not candidates:
            raise FileNotFoundError(f"No weight file in {path}")
        path = candidates[0]
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path))
    else:
        blob = torch.load(str(path), map_location="cpu", weights_only=False)
        state = blob
        for key in ("state_dict", "model", "encoder", "backbone"):
            if isinstance(state, dict) and key in state and isinstance(state[key], dict):
                state = state[key]
                break
        if not isinstance(state, dict):
            raise ValueError(f"{path} does not hold a state dict")
        state = {str(k): v for k, v in state.items() if torch.is_tensor(v)}
    reference = model.state_dict()
    stripped = {}
    for key, value in state.items():
        mapped = key
        for prefix in ("module.", "model.", "encoder.model.", "encoder."):
            if mapped.startswith(prefix) and mapped[len(prefix):] in reference:
                mapped = mapped[len(prefix):]
                break
        stripped[mapped] = value
    missing, unexpected = model.load_state_dict(stripped, strict=False)
    matched = len(reference) - len(missing)
    if matched < 0.5 * len(reference):
        raise RuntimeError(
            f"{path} matched only {matched}/{len(reference)} tensors; "
            f"first missing: {sorted(missing)[:5]}"
        )
    if unexpected:
        print(f"  note: ignored {len(unexpected)} unexpected tensors from {path.name}")
