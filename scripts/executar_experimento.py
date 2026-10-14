#!/usr/bin/env python3
"""Interface de linha de comando para executar a comparação completa."""

from __future__ import annotations

import argparse
import sys

from comparacao import executar_comparacao


class _AnalisadorArgumentos(argparse.ArgumentParser):
    """Traduz os textos fixos produzidos automaticamente pelo argparse."""

    def format_usage(self) -> str:
        return super().format_usage().replace("usage:", "uso:", 1)

    def format_help(self) -> str:
        return (
            super()
            .format_help()
            .replace("usage:", "uso:", 1)
            .replace("options:", "opções:", 1)
        )

    def error(self, mensagem: str) -> None:
        mensagem = mensagem.replace(
            "the following arguments are required:",
            "os seguintes argumentos são obrigatórios:",
        ).replace("unrecognized arguments:", "argumentos não reconhecidos:")
        self.print_usage(sys.stderr)
        self.exit(2, f"{self.prog}: erro: {mensagem}\n")


def principal() -> None:
    analisador = _AnalisadorArgumentos(
        description="Executa a comparação do CREDO.",
        add_help=False,
    )
    analisador.add_argument(
        "-h", "--ajuda", action="help", help="mostra esta ajuda e encerra"
    )
    analisador.add_argument(
        "--configuracao",
        required=True,
        metavar="ARQUIVO",
        help="arquivo YAML com os parâmetros da comparação",
    )
    analisador.add_argument(
        "--diretorio-dados",
        default="dados/processados",
        metavar="DIRETORIO",
        help="diretório com os conjuntos processados",
    )
    analisador.add_argument(
        "--diretorio-resultados",
        default="resultados",
        metavar="DIRETORIO",
        help="diretório no qual os resultados serão salvos",
    )
    analisador.add_argument(
        "--sem-catboost",
        action="store_true",
        help="executa somente os métodos baseados em QNN",
    )
    argumentos = analisador.parse_args()
    saida = executar_comparacao(
        argumentos.configuracao,
        diretorio_dados=argumentos.diretorio_dados,
        diretorio_resultados=argumentos.diretorio_resultados,
        incluir_catboost=not argumentos.sem_catboost,
    )
    print(f"Resultados salvos em {saida}")

if __name__ == "__main__":
    principal()
