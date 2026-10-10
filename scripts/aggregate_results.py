#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from credal_cp_tgb.evaluation.aggregation import aggregate_metrics

parser = argparse.ArgumentParser()
parser.add_argument("metrics_csv")
parser.add_argument("--output", default=None)
args = parser.parse_args()
source = Path(args.metrics_csv)
output = Path(args.output) if args.output else source.with_name("summary.csv")
aggregate_metrics(pd.read_csv(source)).to_csv(output, index=False)
print(output)

