"""Manifest and corpus loaders for foundation calibration and downstream tasks.

Each downstream dataset returns ``(waveform [T], length, label)``.
Collate functions pad to the batch maximum and preserve true lengths for masking.
"""

from __future__ import annotations

import csv
import os
import random
import wave
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 16_000
KWS_LABELS = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
IEMOCAP_LABEL_MAP = {"ang": 0, "hap": 1, "exc": 1, "sad": 2, "neu": 3}
IEMOCAP_CLASSES = ["angry", "happy", "sad", "neutral"]


# ---------------------------------------------------------------------------
# Shared audio helpers
# ---------------------------------------------------------------------------


def load_mono(path: str, sample_rate: int, resamplers: dict) -> torch.Tensor:
    waveform, sr = torchaudio.load(path)
    if waveform.size(0) > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    waveform = waveform.squeeze(0)
    if sr != sample_rate:
        if sr not in resamplers:
            resamplers[sr] = torchaudio.transforms.Resample(sr, sample_rate)
        waveform = resamplers[sr](waveform.unsqueeze(0)).squeeze(0)
    return waveform.clamp(-1.0, 1.0)


def _audio_duration_seconds(path: str | Path) -> float:
    """Read WAV duration from the header, falling back to torchaudio if needed."""
    try:
        with wave.open(str(path), "rb") as handle:
            rate = int(handle.getframerate())
            if rate <= 0:
                raise RuntimeError(f"Could not determine sample rate for {path}")
            return float(handle.getnframes()) / float(rate)
    except (wave.Error, EOFError):
        waveform, sr = torchaudio.load(str(path))
        if int(sr) <= 0:
            raise RuntimeError(f"Could not determine sample rate for {path}")
        return float(waveform.size(-1)) / float(sr)

def pad_or_trim(waveform: torch.Tensor, max_len: int) -> torch.Tensor:
    if waveform.numel() > max_len:
        return waveform[:max_len]
    if waveform.numel() < max_len:
        return F.pad(waveform, (0, max_len - waveform.numel()))
    return waveform


def classification_collate(batch):
    waveforms, lengths, labels = zip(*batch)
    max_len = max(int(L) for L in lengths)
    padded = []
    for wave, length in zip(waveforms, lengths):
        wave = wave[: int(length)]
        if wave.numel() < max_len:
            wave = F.pad(wave, (0, max_len - wave.numel()))
        padded.append(wave)
    return {
        "waveforms": torch.stack(padded, dim=0),
        "lengths": torch.tensor(lengths, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def foundation_collate(batch):
    waveforms, lengths = zip(*batch)
    max_len = max(int(L) for L in lengths)
    padded = []
    for wave, length in zip(waveforms, lengths):
        wave = wave[: int(length)]
        if wave.numel() < max_len:
            wave = F.pad(wave, (0, max_len - wave.numel()))
        padded.append(wave)
    return {
        "waveforms": torch.stack(padded, dim=0),
        "lengths": torch.tensor(lengths, dtype=torch.long),
    }



# ---------------------------------------------------------------------------
# Common Voice (unlabelled foundation / sigma calibration)
# ---------------------------------------------------------------------------


class CommonVoiceManifestDataset(Dataset):
    def __init__(
        self,
        root: str,
        manifest: str,
        sample_rate: int = SAMPLE_RATE,
        max_len_sec: float = 10.0,
        max_samples: int | None = None,
        seed: int = 42,
    ):
        self.root = Path(root)
        self.clips = self.root / "clips"
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_len_sec) if max_len_sec else None
        self.seed = seed
        self._resamplers: dict = {}
        frame = pd.read_csv(manifest, sep="\t", low_memory=False)
        col = "path" if "path" in frame.columns else ("clip" if "clip" in frame.columns else frame.columns[1])
        names = frame[col].dropna().astype(str).map(lambda p: Path(p).name).tolist()
        if max_samples is not None and int(max_samples) < len(names):
            rng = random.Random(seed)
            names = rng.sample(names, int(max_samples))
        self.paths = [self.clips / name if (self.clips / name).is_file() else self.root / name for name in names]
        missing = [p for p in self.paths if not p.is_file()]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} Common Voice clips missing; first: {missing[0]}"
            )

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        waveform = load_mono(str(self.paths[idx]), self.sample_rate, self._resamplers)
        if self.max_len is not None:
            waveform = waveform[: self.max_len]
        return waveform, int(waveform.numel())



# ---------------------------------------------------------------------------
# Keyword spotting — Speech Commands v0.02
# ---------------------------------------------------------------------------


def _speech_commands_walker(corpus_root: Path, subset: str) -> list[Path]:
    if subset not in {"training", "validation", "testing"}:
        raise ValueError(f"Unknown Speech Commands subset: {subset!r}")

    validation = [
        line.strip() for line in (corpus_root / "validation_list.txt").read_text().splitlines() if line.strip()
    ]
    testing = [
        line.strip() for line in (corpus_root / "testing_list.txt").read_text().splitlines() if line.strip()
    ]
    held_out = set(validation) | set(testing)
    if subset == "validation":
        relative = validation
    elif subset == "testing":
        relative = testing
    else:
        relative = [
            p.relative_to(corpus_root).as_posix()
            for p in sorted(corpus_root.glob("*/*.wav"))
            if p.parent.name != "_background_noise_"
            and p.relative_to(corpus_root).as_posix() not in held_out
        ]
    paths = [corpus_root / rel for rel in relative]
    missing = [p for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"Speech Commands {subset} missing {len(missing)} files; first: {missing[0]}")
    return paths


class SpeechCommandsKWS(Dataset):
    """Speech Commands v0.02 with ten keywords, unknown, and silence."""

    def __init__(
        self,
        root: str,
        subset: str,
        label_set=None,
        sample_rate: int = SAMPLE_RATE,
        max_len_sec: float = 1.0,
        include_silence: bool = True,
        unknown_percentage: float = 10.0,
        silence_percentage: float = 10.0,
        sampling_seed: int = 59185,
        crop_seed: int = 20260905,
    ):
        if subset not in {"training", "validation", "testing"}:
            raise ValueError(f"Unknown Speech Commands subset: {subset!r}")
        root = Path(root).expanduser()
        nested = root / "SpeechCommands" / "speech_commands_v0.02"
        if (root / "validation_list.txt").is_file():
            corpus = root
        elif (nested / "validation_list.txt").is_file():
            corpus = nested
        else:
            raise FileNotFoundError(f"Speech Commands v0.02 not found under {root}")
        self._path = corpus
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_len_sec)
        self.crop_seed = int(crop_seed)
        self.label_set = list(label_set or KWS_LABELS)
        self.classes = self.label_set + ["unknown"] + (["silence"] if include_silence else [])
        self.label2idx = {name: i for i, name in enumerate(self.classes)}
        self._resamplers: dict = {}

        # Match the reference recipe's deterministic unknown sampling order.
        by_split = {}
        for name in ("validation", "testing", "training"):
            paths = _speech_commands_walker(corpus, name)
            by_split[name] = {
                "wanted": [p for p in paths if p.parent.name in self.label_set],
                "unknown": [p for p in paths if p.parent.name not in self.label_set],
            }
        if not by_split[subset]["wanted"]:
            raise RuntimeError(f"Speech Commands {subset} has no wanted-word examples")

        self.unknown_percentage = float(unknown_percentage)
        self.silence_percentage = float(silence_percentage) if include_silence else 0.0
        rng = random.Random(int(sampling_seed))
        selected_by_split = {}
        for name in ("validation", "testing", "training"):
            wanted_count = len(by_split[name]["wanted"])
            pool = list(by_split[name]["unknown"])
            target = int(np.ceil(wanted_count * self.unknown_percentage / 100.0))
            if target > len(pool):
                raise RuntimeError(
                    f"Speech Commands {name}: need {target} unknown examples for "
                    f"{self.unknown_percentage:g}% balancing, but only {len(pool)} are available"
                )
            rng.shuffle(pool)
            selected_by_split[name] = pool[:target]

        wanted = by_split[subset]["wanted"]
        selected_unknown = selected_by_split[subset]
        self.num_wanted = len(wanted)
        self.num_unknown = len(selected_unknown)
        # Wanted examples are all retained; unknown is the balanced subset.
        self._walker = wanted + selected_unknown

        self.silence_clips: list[str] = []
        self._n_silence = 0
        if include_silence:
            noise_dir = corpus / "_background_noise_"
            if noise_dir.is_dir():
                self.silence_clips = [str(noise_dir / f) for f in sorted(os.listdir(noise_dir)) if f.endswith(".wav")]
            if not self.silence_clips:
                raise FileNotFoundError(
                    f"Speech Commands {subset}: silence balancing requested but no _background_noise_/*.wav files found"
                )
            # Whole-recording holdout: lexicographically first recording is
            # validation, second is testing, all remaining recordings train.
            # No source sample crosses splits, even when a short clip is padded.
            # Require three sources instead of silently sharing audio/seeds.
            if len(self.silence_clips) < 3:
                raise ValueError("Disjoint silence requires at least three background recordings")
            allocation = {
                "validation": self.silence_clips[:1],
                "testing": self.silence_clips[1:2],
                "training": self.silence_clips[2:],
            }
            self.silence_clips = allocation[subset]
            self._n_silence = int(np.ceil(self.num_wanted * self.silence_percentage / 100.0))

    def __len__(self) -> int:
        return len(self._walker) + self._n_silence

    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def __getitem__(self, idx: int):
        n_main = len(self._walker)
        if idx < n_main:
            path = self._walker[int(idx)]
            waveform = load_mono(str(path), self.sample_rate, self._resamplers)
            spoken = path.parent.name
            label_name = spoken if spoken in self.label_set else "unknown"
            length = min(waveform.numel(), self.max_len)
            waveform = pad_or_trim(waveform, self.max_len)
            return waveform, length, self.label2idx[label_name]
        silence_idx = idx - n_main
        clip_idx = silence_idx % len(self.silence_clips)
        waveform = load_mono(self.silence_clips[clip_idx], self.sample_rate, self._resamplers)
        if waveform.numel() > self.max_len:
            span = waveform.numel() - self.max_len + 1
            generator = torch.Generator().manual_seed(self.crop_seed + int(silence_idx))
            start = int(torch.randint(0, span, (1,), generator=generator).item())
            waveform = waveform[start:start + self.max_len]
        waveform = pad_or_trim(waveform, self.max_len)
        return waveform, self.max_len, self.label2idx["silence"]


# ---------------------------------------------------------------------------
# Intent classification — Fluent Speech Commands
# ---------------------------------------------------------------------------


def _intent_key(action: str, obj: str, location: str) -> str:
    return f"{action}|{obj}|{location}"


def build_intent_vocab(root_dir: str) -> dict[str, int]:
    train_csv = Path(root_dir) / "data" / "train_data.csv"
    intents = set()
    with open(train_csv, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            intents.add(_intent_key(row["action"], row["object"], row["location"]))
    return {name: i for i, name in enumerate(sorted(intents))}


class FluentSpeechCommands(Dataset):
    def __init__(
        self,
        root: str,
        split: str,
        sample_rate: int = SAMPLE_RATE,
        max_len_sec: float = 15.0,
        intent_to_idx: dict[str, int] | None = None,
    ):
        if split not in {"train", "valid", "test"}:
            raise ValueError(f"FSC split must be train/valid/test, got {split!r}")
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_len_sec) if max_len_sec else None
        self._resamplers: dict = {}
        self.intent_to_idx = intent_to_idx or build_intent_vocab(str(self.root))
        self.classes = [name for name, _ in sorted(self.intent_to_idx.items(), key=lambda kv: kv[1])]
        self.num_classes = len(self.classes)
        csv_path = self.root / "data" / f"{split}_data.csv"
        if not csv_path.is_file():
            raise FileNotFoundError(csv_path)
        samples = []
        with open(csv_path, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                rel = row["path"].lstrip("./")
                wav_path = self.root / rel
                intent = _intent_key(row["action"], row["object"], row["location"])
                if intent not in self.intent_to_idx:
                    raise ValueError(f"Intent {intent!r} in {csv_path} is absent from the training vocabulary")
                if not wav_path.is_file():
                    raise FileNotFoundError(wav_path)
                samples.append((wav_path, self.intent_to_idx[intent]))
        self.samples = samples
        if not self.samples:
            raise RuntimeError(f"No Fluent Speech Commands utterances in {csv_path}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        waveform = load_mono(str(path), self.sample_rate, self._resamplers)
        if self.max_len is not None:
            waveform = waveform[: self.max_len]
        return waveform, int(waveform.numel()), int(label)


# ---------------------------------------------------------------------------
# Speaker identification — VoxCeleb1 / SID manifests
# ---------------------------------------------------------------------------


class SpeakerIDDataset(Dataset):
    """CSV manifests ``{split}_manifest.csv`` with ``file_path`` and ``speaker_id``/``label``.

    Training utterances above the configured duration are discarded; validation
    and test utterances are kept in full.
    """

    def __init__(
        self,
        root: str,
        subset: str = "train",
        sample_rate: int = SAMPLE_RATE,
        train_max_duration_sec: float | None = 10.0,
        drop_long_train: bool = True,
        seed: int = 42,
        label_to_index: dict | None = None,
        max_samples: int | None = None,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.seed = seed
        self._resamplers: dict = {}
        aliases = {
            "train": ["train", "training"],
            "val": ["val", "validation", "dev"],
            "test": ["test", "testing"],
        }
        manifest = None
        for name in aliases.get(subset, [subset]):
            candidate = self.root / f"{name}_manifest.csv"
            if candidate.is_file():
                manifest = candidate
                break
        if manifest is None:
            raise FileNotFoundError(f"No SID manifest for split {subset!r} under {root}")
        frame = pd.read_csv(manifest)
        if "label" in frame.columns and "speaker_id" not in frame.columns:
            frame = frame.rename(columns={"label": "speaker_id"})
        frame["full_path"] = frame["file_path"].map(lambda p: str(self.root / p) if not os.path.isabs(str(p)) else str(p))
        if max_samples is not None and int(max_samples) < len(frame):
            selected = np.random.RandomState(seed).choice(len(frame), int(max_samples), replace=False)
            frame = frame.iloc[np.sort(selected)].reset_index(drop=True)
        self.dropped_long_train = 0
        if subset in {"train", "training"} and drop_long_train and train_max_duration_sec is not None:
            keep = []
            limit = float(train_max_duration_sec)
            for path in frame["full_path"].tolist():
                keep.append(_audio_duration_seconds(path) <= limit + 1e-9)
            self.dropped_long_train = len(keep) - sum(keep)
            frame = frame.loc[keep].reset_index(drop=True)
        speakers = sorted(frame["speaker_id"].astype(str).unique().tolist())
        self.label2idx = dict(label_to_index) if label_to_index is not None else {s: i for i, s in enumerate(speakers)}
        self.classes = [s for s, _ in sorted(self.label2idx.items(), key=lambda kv: kv[1])]
        self.num_classes = len(self.classes)
        self.frame = frame.reset_index(drop=True)
        if self.dropped_long_train:
            print(f"SID train: discarded {self.dropped_long_train} utterances longer than {float(train_max_duration_sec):g} s")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        row = self.frame.iloc[idx]
        waveform = load_mono(row["full_path"], self.sample_rate, self._resamplers)
        label = self.label2idx[str(row["speaker_id"])]
        return waveform, int(waveform.numel()), int(label)


class VoxCeleb1Identification(Dataset):
    """Official VoxCeleb1 ``iden_split.txt`` speaker-identification protocol.

    Split codes 1/2/3 are train/validation/test. Only long training files are
    filtered; validation and test utterances are left untouched.
    """

    SPLIT_CODE = {"train": 1, "training": 1, "val": 2, "validation": 2, "dev": 2, "test": 3, "testing": 3}

    def __init__(
        self,
        root: str,
        subset: str = "train",
        sample_rate: int = SAMPLE_RATE,
        train_max_duration_sec: float | None = 10.0,
        drop_long_train: bool = True,
        seed: int = 42,
        speaker2idx: dict | None = None,
        num_speakers: int | None = None,
    ):
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.seed = seed
        self._resamplers: dict = {}
        iden = self.root / "iden_split.txt"
        if not iden.is_file():
            raise FileNotFoundError(f"VoxCeleb1 iden_split.txt not found in {root}")
        if subset not in self.SPLIT_CODE:
            raise ValueError(f"Unknown VoxCeleb1 split: {subset!r}")
        code = self.SPLIT_CODE[subset]
        samples: list[tuple[Path, str]] = []
        speakers: set[str] = set()
        missing: list[str] = []
        for line in iden.read_text().splitlines():
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                split_code = int(parts[0])
            except ValueError:
                continue
            rel = parts[1]
            speaker = rel.split("/")[0]
            speakers.add(speaker)
            if split_code != code:
                continue
            for candidate in (self.root / rel, self.root / "wav" / rel):
                if candidate.is_file():
                    samples.append((candidate, speaker))
                    break
            else:
                missing.append(rel)
        if missing:
            raise FileNotFoundError(
                f"VoxCeleb1 {subset}: {len(missing)} files from iden_split.txt are missing; "
                f"first: {missing[0]}"
            )
        self.dropped_long_train = 0
        if subset in {"train", "training"} and drop_long_train and train_max_duration_sec is not None:
            limit = float(train_max_duration_sec)
            filtered = []
            for path, speaker in samples:
                if _audio_duration_seconds(path) <= limit + 1e-9:
                    filtered.append((path, speaker))
                else:
                    self.dropped_long_train += 1
            samples = filtered
            if self.dropped_long_train:
                print(f"SID train: discarded {self.dropped_long_train} utterances longer than {limit:g} s")
        if speaker2idx is None:
            ordered = sorted(speakers)
            if num_speakers:
                ordered = ordered[: int(num_speakers)]
            speaker2idx = {s: i for i, s in enumerate(ordered)}
        allowed = set(speaker2idx)
        self.speaker2idx = speaker2idx
        self.classes = [s for s, _ in sorted(speaker2idx.items(), key=lambda kv: kv[1])]
        self.num_classes = len(self.classes)
        self.samples = [(p, s) for p, s in samples if s in allowed]
        if not self.samples:
            raise RuntimeError(f"No VoxCeleb1 files for split {subset!r} under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, speaker = self.samples[idx]
        waveform = load_mono(str(path), self.sample_rate, self._resamplers)
        return waveform, int(waveform.numel()), int(self.speaker2idx[speaker])


# ---------------------------------------------------------------------------
# Emotion recognition — IEMOCAP (leave-one-session-out)
# ---------------------------------------------------------------------------


class IEMOCAPDataset(Dataset):
    def __init__(
        self,
        root: str,
        subset: str = "train",
        fold: int = 5,
        sample_rate: int = SAMPLE_RATE,
        max_len_sec: float = 10.0,
        validation_fraction: float = 0.10,
        validation_seed: int = 0,
    ):
        if subset not in {"train", "val", "test"}:
            raise ValueError(subset)
        if not 1 <= int(fold) <= 5:
            raise ValueError("IEMOCAP fold must be 1..5")
        self.root = Path(root)
        self.sample_rate = sample_rate
        self.max_len = int(sample_rate * max_len_sec) if max_len_sec else None
        self._resamplers: dict = {}
        self.classes = list(IEMOCAP_CLASSES)
        self.num_classes = len(self.classes)
        held = int(fold)
        if subset == "test":
            sessions = [held]
        else:
            sessions = [s for s in (1, 2, 3, 4, 5) if s != held]
        samples: list[tuple[Path, int]] = []
        missing_audio: list[Path] = []
        import re
        pattern = re.compile(r"^\[\d+\.\d+\s*-\s*\d+\.\d+\]\s+(\S+)\s+([a-z]+)")
        for sess in sessions:
            sess_dir = self.root / f"Session{sess}"
            label_dir = sess_dir / "dialog" / "EmoEvaluation"
            wav_root = sess_dir / "sentences" / "wav"
            if not label_dir.is_dir():
                raise FileNotFoundError(f"IEMOCAP labels missing: {label_dir}")
            for label_file in sorted(label_dir.glob("*.txt")):
                dialog_id = label_file.stem
                wav_dir = wav_root / dialog_id
                for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
                    match = pattern.match(line.strip())
                    if not match:
                        continue
                    utt_id, emotion = match.group(1), match.group(2)
                    if emotion not in IEMOCAP_LABEL_MAP:
                        continue
                    audio = wav_dir / f"{utt_id}.wav"
                    if audio.is_file():
                        samples.append((audio, IEMOCAP_LABEL_MAP[emotion]))
                    else:
                        missing_audio.append(audio)
        if missing_audio:
            raise FileNotFoundError(
                f"IEMOCAP: {len(missing_audio)} annotated utterances are missing; "
                f"first: {missing_audio[0]}"
            )
        if subset in {"train", "val"}:
            rng = np.random.RandomState(int(validation_seed))
            order = rng.permutation(len(samples))
            cut = max(1, int(float(validation_fraction) * len(samples)))
            if subset == "val":
                samples = [samples[i] for i in sorted(order[:cut])]
            else:
                keep = set(order[cut:].tolist())
                samples = [samples[i] for i in range(len(samples)) if i in keep]
        self.samples = samples
        if not self.samples:
            raise RuntimeError(f"No IEMOCAP samples for subset={subset} fold={fold}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        waveform = load_mono(str(path), self.sample_rate, self._resamplers)
        if self.max_len is not None:
            waveform = waveform[: self.max_len]
        return waveform, int(waveform.numel()), int(label)




def build_foundation_loader(cfg: dict, manifest_key: str, samples_key: str, shuffle: bool) -> DataLoader:
    data = cfg["data"]
    foundation = cfg.get("foundation", {})
    max_samples = data.get(samples_key)
    if manifest_key == "sigma_manifest" and max_samples is None:
        max_samples = foundation.get("sigma_samples")
    dataset = CommonVoiceManifestDataset(
        data["root"],
        data[manifest_key],
        sample_rate=int(data.get("sample_rate", SAMPLE_RATE)),
        max_len_sec=float(data.get("max_len_sec", 10.0)),
        max_samples=max_samples,
        seed=int(cfg.get("seed", 42)),
    )
    batch_size = int(foundation.get("sigma_batch_size", cfg.get("train", {}).get("batch_size", 1)))
    if manifest_key != "sigma_manifest":
        batch_size = int(cfg["train"].get("batch_size", 1))
    workers = int(foundation.get("sigma_num_workers", cfg.get("train", {}).get("num_workers", 2)))
    if manifest_key != "sigma_manifest":
        workers = int(cfg["train"].get("num_workers", 2))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        collate_fn=foundation_collate,
    )


def _sid_dataset(data: dict, split: str, label_to_index=None):
    root = data["root"]
    kwargs = dict(
        root=root,
        subset=split,
        sample_rate=int(data.get("sample_rate", SAMPLE_RATE)),
        train_max_duration_sec=(
            None if data.get("train_max_duration_sec") is None
            else float(data.get("train_max_duration_sec", 10.0))
        ),
        drop_long_train=bool(data.get("drop_long_train", True)),
        seed=int(data.get("seed", 42)),
    )
    split_source = str(data.get("split_source", "auto")).lower()
    use_official = split_source == "official_iden_split" or (
        split_source == "auto"
        and (Path(root) / "iden_split.txt").is_file()
        and not (Path(root) / f"{split}_manifest.csv").is_file()
    )
    if use_official:
        if not (Path(root) / "iden_split.txt").is_file():
            raise FileNotFoundError(f"SID split_source=official_iden_split but {Path(root) / 'iden_split.txt'} is missing")
        kwargs["speaker2idx"] = label_to_index
        kwargs["num_speakers"] = data.get("num_speakers")
        return VoxCeleb1Identification(**kwargs)
    if split_source not in {"auto", "manifests"}:
        raise ValueError(f"Unknown SID split_source={split_source!r}")
    kwargs["label_to_index"] = label_to_index
    kwargs["max_samples"] = data.get(f"{split}_samples")
    return SpeakerIDDataset(**kwargs)


def build_task_dataset(cfg: dict, split: str, extra=None):
    task = cfg["task"].lower()
    data = cfg["data"]
    extra = extra or {}
    sample_rate = int(data.get("sample_rate", SAMPLE_RATE))
    if task in {"ks", "kws"}:
        subset = {"train": "training", "val": "validation", "test": "testing"}[split]
        return SpeechCommandsKWS(
            root=data["root"],
            subset=subset,
            label_set=data.get("label_set"),
            sample_rate=sample_rate,
            max_len_sec=float(data.get("max_len_sec", 1.0)),
            include_silence=bool(data.get("include_silence", True)),
            unknown_percentage=float(data.get("unknown_percentage", 10.0)),
            silence_percentage=float(data.get("silence_percentage", 10.0)),
            sampling_seed=int(data.get("sampling_seed", 59185)),
        )
    if task in {"ic", "intent"}:
        subset = {"train": "train", "val": "valid", "test": "test"}[split]
        return FluentSpeechCommands(
            root=data["root"],
            split=subset,
            sample_rate=sample_rate,
            max_len_sec=float(data.get("max_len_sec", 15.0)),
            intent_to_idx=extra.get("intent_to_idx"),
        )
    if task in {"sid", "si"}:
        subset = {"train": "train", "val": "val", "test": "test"}[split]
        return _sid_dataset(data, subset, extra.get("label_to_index"))
    if task in {"er", "ser", "iemocap"}:
        return IEMOCAPDataset(
            root=data["root"],
            subset=split,
            fold=int(data.get("fold", 5)),
            sample_rate=sample_rate,
            max_len_sec=float(data.get("max_len_sec", 10.0)),
            validation_fraction=float(data.get("validation_fraction", 0.10)),
            validation_seed=int(data.get("validation_seed", 0)),
        )
    raise ValueError(f"Unknown task {task!r}")


def build_classification_loaders(cfg: dict) -> tuple[Dataset, Dataset, Dataset, DataLoader, DataLoader, DataLoader]:
    task = cfg["task"].lower()
    extra = {}
    if task in {"ic", "intent"}:
        extra["intent_to_idx"] = build_intent_vocab(cfg["data"]["root"])
    train_set = build_task_dataset(cfg, "train", extra)
    if task in {"sid", "si"}:
        extra["label_to_index"] = getattr(train_set, "label2idx", getattr(train_set, "speaker2idx", None))
    val_set = build_task_dataset(cfg, "val", extra)
    test_set = build_task_dataset(cfg, "test", extra)
    collate = classification_collate
    loader_kwargs = dict(
        batch_size=int(cfg["data"].get("batch_size", cfg.get("adapt", {}).get("feature_batch_size", 8))),
        num_workers=int(cfg["data"].get("num_workers", 4)),
        collate_fn=collate,
    )
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_set, val_set, test_set, train_loader, val_loader, test_loader
