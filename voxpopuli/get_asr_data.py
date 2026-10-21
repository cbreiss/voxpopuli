# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import csv
import argparse
import os
from tqdm import tqdm
from ast import literal_eval
import gzip
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict

import numpy as np
import soundfile as sf
from torchaudio.datasets.utils import download_url  # patched by src._torchaudio_compat

from voxpopuli import ASR_LANGUAGES, ASR_ACCENTED_LANGUAGES, DOWNLOAD_BASE_URL
from voxpopuli.utils import multiprocess_run


SPLITS = ["train", "dev", "test"]


def cut_session(info: Tuple[str, Dict[str, List[Tuple[float, float]]]]) -> None:
    in_path, out_path_to_timestamps = info
    if not Path(in_path).exists():
        # Source session not on disk (partial download — e.g. --max-years); skip silently.
        return
    # Skip sessions where every output segment already exists (idempotent re-runs).
    if all(Path(p).exists() for p in out_path_to_timestamps):
        return
    # soundfile returns (samples, channels); torchaudio was (channels, samples).
    # We work in (samples, channels) throughout and write with sf.write.
    waveform, sr = sf.read(in_path, always_2d=True)
    duration = waveform.shape[0]
    for out_path, timestamps in out_path_to_timestamps.items():
        segment = np.concatenate(
            [waveform[int(s * sr): min(int(t * sr), duration), :]
             for s, t in timestamps],
            axis=0,
        )
        # Write atomically: the skip-check above treats "every segment exists"
        # as "session done", so a worker killed mid-write (e.g. OOM) must never
        # leave a truncated file under the final name.  The ".part" name is not
        # matched by *.ogg globs, and the format is given explicitly because it
        # can no longer be inferred from the extension.
        tmp_path = f"{out_path}.part"
        sf.write(tmp_path, segment, sr, format="OGG")
        os.replace(tmp_path, out_path)


def get(args):
    in_root = Path(args.root) / "raw_audios" / "original"
    out_root = Path(args.root) / "transcribed_data" / args.lang
    out_root.mkdir(exist_ok=True, parents=True)
    # Get metadata TSV
    url = f"{DOWNLOAD_BASE_URL}/annotations/asr/asr_{args.lang}.tsv.gz"
    tsv_path = out_root / Path(url).name
    if not tsv_path.exists():
        download_url(url, out_root.as_posix(), Path(url).name)
    with gzip.open(tsv_path, "rt", encoding="utf-8") as f:
        metadata = [x for x in csv.DictReader(f, delimiter="|")]
    # Get segment into list
    items = defaultdict(dict)
    manifest = []
    for r in tqdm(metadata):
        split = r["split"]
        if split not in SPLITS:
            continue
        event_id = r["session_id"]
        year = event_id[:4]
        in_path = in_root / year / f"{event_id}_original.ogg"
        cur_out_root = out_root / year
        cur_out_root.mkdir(exist_ok=True, parents=True)
        # Sanitize id_ for Windows: colons (e.g. in HH:MM:SS timestamps) are
        # illegal in NTFS filenames; replace with hyphens.
        safe_id = r["id_"].replace(":", "-")
        out_path = cur_out_root / "{}-{}.ogg".format(event_id, safe_id)
        timestamps = [(t[0], t[1]) for t in literal_eval(r["vad"])]
        items[in_path.as_posix()][out_path.as_posix()] = timestamps
        manifest.append(
            (
             out_path.stem,
             r["original_text"],
             r["normed_text"],
             r["speaker_id"],
             split,
             r["gender"],
             r.get("is_gold_transcript", str(False)),
             r.get("accent", str(None))
            )
        )
    items = list(items.items())
    # Segment
    # Pool size: args.n_workers if given (the caller's CPU allocation), else
    # tqdm's default of min(32, cpu_count + 4), which counts the whole node.
    # Each worker holds one whole decoded session (full-day sessions reach
    # ~15 h ≈ 6.8 GB as float64), so an unbounded pool on a large node can
    # exceed the job's memory limit and get a worker OOM-killed.
    multiprocess_run(items, cut_session, n_workers=getattr(args, "n_workers", None))
    # Output per-split manifest
    header = [
        "id", "raw_text", "normalized_text", "speaker_id", "split",
        "gender", "is_gold_transcript", "accent"
    ]
    for split in SPLITS:
        with open(out_root / f"asr_{split}.tsv", "w", encoding="utf-8") as f_o:
            f_o.write("\t".join(header) + "\n")
            for cols in manifest:
                if cols[4] == split:
                    f_o.write("\t".join(cols) + "\n")


def get_args():
    parser = argparse.ArgumentParser("Prepare transcribed data")
    parser.add_argument(
        "--root",
        help="data root path",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--lang",
        required=True,
        type=str,
        choices=ASR_LANGUAGES + ASR_ACCENTED_LANGUAGES,
    )
    return parser.parse_args()


def main():
    args = get_args()
    get(args)


if __name__ == "__main__":
    main()
