#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from credal_cp_tgb.data.download import DATASET_URLS, download_dataset


parser = argparse.ArgumentParser()
parser.add_argument("dataset", choices=sorted(DATASET_URLS))
parser.add_argument("--destination", default=None)
args = parser.parse_args()
destination = args.destination or f"data/raw/{args.dataset}/{Path(DATASET_URLS[args.dataset]).name}"
print(download_dataset(args.dataset, destination))

