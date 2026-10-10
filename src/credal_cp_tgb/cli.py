from __future__ import annotations

import argparse

from .benchmark import run_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description="Executa o benchmark do CREDO.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--without-catboost", action="store_true")
    args = parser.parse_args()
    output = run_benchmark(
        args.config,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        include_catboost=not args.without_catboost,
    )
    print(f"Resultados salvos em {output}")


if __name__ == "__main__":
    main()

