"""Gera resumos consolidados e tabelas formatadas dos experimentos reais."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from credal_cp_tgb.avaliacao.agregacao import agregar_metricas

ORDEM_CONJUNTOS = [
    "concrete",
    "airfoil",
    "winewhite",
    "star",
    "winered",
    "cycle",
    "electric",
    "meps19",
    "superconductivity",
    "homes",
    "protein",
    "WEC",
]

ORDEM_METODOS = [
    "CREDO-QNN",
    "CREDO-QNN-Adaptativo",
    "CQR-QNN",
    "CQR-r-QNN",
    "CQR-CatBoost",
    "CQR-r-CatBoost",
    "UACQR-S-CatBoost",
    "UACQR-P-CatBoost",
    "EPICScore-CatBoost-MDN",
]

NOMES_METODOS = {
    "CREDO-QNN": "CREDO",
    "CREDO-QNN-Adaptativo": "CREDO (adap.)",
    "CQR-QNN": "CQR (QNN)",
    "CQR-r-QNN": "CQR-r (QNN)",
    "CQR-CatBoost": "CQR (CatBoost)",
    "CQR-r-CatBoost": "CQR-r (CatBoost)",
    "UACQR-S-CatBoost": "UACQR-S",
    "UACQR-P-CatBoost": "UACQR-P",
    "EPICScore-CatBoost-MDN": "EPIC",
}

METRICAS = {
    "smis": ("SMIS", 2, "minimo"),
    "il": ("Comprimento médio do intervalo", 2, "minimo"),
    "pcorr": ("PCorr", 3, "minimo"),
    "ilr": ("ILR", 2, "maximo"),
    "amc": ("Cobertura marginal", 3, None),
    "aco": ("Cobertura nos valores atípicos", 3, "alvo"),
    "segundos_total_metodo": ("Tempo total médio (s)", 2, "minimo"),
}

NOMES_ARQUIVOS = {
    "segundos_total_metodo": "tempo",
}

ESCALAS_SMIS = {
    "star": 1e2,
    "electric": 1e-2,
    "homes": 1e4,
    "WEC": 1e4,
}

ROTULOS_ESCALA = {
    "star": "star ×10²",
    "electric": "electric ×10⁻²",
    "homes": "homes ×10⁴",
    "WEC": "WEC ×10⁴",
}


def _marcar_melhores(dados: pd.DataFrame, criterio: str | None) -> pd.Series:
    """Marca melhores médias, permitindo empates pelos ICs de 95%."""
    destaques = pd.Series(False, index=dados.index)
    if criterio is None:
        return destaques

    for _, grupo in dados.groupby("conjunto"):
        inferior = grupo["media"] - 1.96 * grupo["erro_padrao"]
        superior = grupo["media"] + 1.96 * grupo["erro_padrao"]
        if criterio == "alvo":
            contem_alvo = (inferior <= 0.90) & (superior >= 0.90)
            if contem_alvo.any():
                destaques.loc[grupo.index] = contem_alvo
            else:
                distancia = (grupo["media"] - 0.90).abs()
                destaques.loc[distancia.idxmin()] = True
            continue

        indice_melhor = (
            grupo["media"].idxmin() if criterio == "minimo" else grupo["media"].idxmax()
        )
        limite_inferior_melhor = inferior.loc[indice_melhor]
        limite_superior_melhor = superior.loc[indice_melhor]
        intervalos_sobrepostos = (
            (inferior <= limite_superior_melhor)
            & (superior >= limite_inferior_melhor)
        )
        destaques.loc[grupo.index] = intervalos_sobrepostos
    return destaques


def formatar_tabela(
    resumo: pd.DataFrame,
    metrica: str,
    casas: int,
    criterio: str | None,
    *,
    destacar: bool = False,
) -> pd.DataFrame:
    dados = resumo[resumo["metrica"] == metrica].copy()
    dados["destaque"] = _marcar_melhores(dados, criterio)
    if metrica == "smis":
        escala = dados["conjunto"].map(ESCALAS_SMIS).fillna(1.0)
        dados["media"] = dados["media"] / escala
        dados["erro_padrao"] = dados["erro_padrao"] / escala
    dados["valor"] = dados.apply(
        lambda linha: f"{linha['media']:.{casas}f} ({linha['erro_padrao']:.{casas}f})",
        axis=1,
    )
    if destacar:
        dados.loc[dados["destaque"], "valor"] = (
            "**" + dados.loc[dados["destaque"], "valor"] + "**"
        )
    tabela = dados.pivot(index="conjunto", columns="metodo", values="valor")
    tabela = tabela.reindex(index=ORDEM_CONJUNTOS, columns=ORDEM_METODOS)
    tabela = tabela.rename(columns=NOMES_METODOS)
    if metrica == "smis":
        tabela = tabela.rename(index=ROTULOS_ESCALA)
    tabela.index.name = "Dataset"
    return tabela


def _texto_latex(valor: object) -> str:
    texto = str(valor)
    texto = texto.replace("**", "")
    substituicoes = {
        "star ×10²": r"star $\times 10^{2}$",
        "electric ×10⁻²": r"electric $\times 10^{-2}$",
        "homes ×10⁴": r"homes $\times 10^{4}$",
        "WEC ×10⁴": r"WEC $\times 10^{4}$",
    }
    return substituicoes.get(texto, texto)


def converter_para_latex(
    tabela: pd.DataFrame,
    *,
    titulo: str,
    metrica: str,
    criterio: str | None,
) -> str:
    """Produz uma tabela LaTeX completa, mantendo os destaques estatísticos."""
    cabecalho = [str(tabela.index.name or "")] + [str(coluna) for coluna in tabela.columns]
    linhas = [
        r"\begin{table*}[htbp]",
        r"\centering",
        rf"\caption{{{titulo}. Cada célula apresenta média (erro-padrão), calculados sobre 30 repetições."
        + (
            " O negrito identifica o melhor resultado e os empates segundo "
            r"intervalos de confiança de 95\%.}"
            if criterio is not None
            else "}"
        ),
        rf"\label{{tab:{metrica}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{l{'c' * len(tabela.columns)}}}",
        r"\toprule",
        " & ".join(cabecalho) + r" \\",
        r"\midrule",
    ]
    for indice, valores in tabela.iterrows():
        celulas = [_texto_latex(indice)]
        for valor in valores.tolist():
            texto = str(valor)
            conteudo = _texto_latex(texto)
            celulas.append(rf"\textbf{{{conteudo}}}" if texto.startswith("**") else conteudo)
        linhas.append(" & ".join(celulas) + r" \\")
    linhas.extend(
        [
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
            r"\end{table*}",
        ]
    )
    return "\n".join(linhas) + "\n"


def main() -> None:
    analisador = argparse.ArgumentParser(
        description="Gera tabelas consolidadas dos experimentos reais."
    )
    analisador.add_argument(
        "--arquivo-metricas",
        action="append",
        type=Path,
        help=(
            "arquivo metricas.csv de origem; pode ser informado mais de uma vez. "
            "Sem esta opção, usa resultados/brutos/resultados_*/metricas.csv"
        ),
    )
    argumentos = analisador.parse_args()

    raiz = Path("resultados")
    arquivos = argumentos.arquivo_metricas or sorted(
        (raiz / "brutos").glob("resultados_*/metricas.csv")
    )
    if not arquivos:
        raise FileNotFoundError("Nenhum arquivo metricas.csv foi encontrado.")

    resultados = pd.concat((pd.read_csv(arquivo) for arquivo in arquivos), ignore_index=True)
    resumo = agregar_metricas(resultados)
    pasta_resumos = raiz / "resumos"
    pasta_tabelas = raiz / "tabelas"
    pasta_resumos.mkdir(parents=True, exist_ok=True)
    pasta_tabelas.mkdir(parents=True, exist_ok=True)
    resumo.to_csv(pasta_resumos / "resumo_completo.csv", index=False)

    linhas_validacao = []
    for (conjunto, metodo), grupo in resultados.groupby(["conjunto", "metodo"]):
        linha = {
            "conjunto": conjunto,
            "metodo": metodo,
            "numero_linhas": len(grupo),
            "numero_repeticoes": grupo["repeticao"].nunique(),
            "numero_duplicatas": int(grupo.duplicated("repeticao").sum()),
        }
        for metrica in METRICAS:
            if metrica not in grupo:
                continue
            linha[f"valores_nao_finitos_{metrica}"] = int(
                (~np.isfinite(pd.to_numeric(grupo[metrica], errors="coerce"))).sum()
            )
        linhas_validacao.append(linha)
    pd.DataFrame(linhas_validacao).to_csv(
        pasta_resumos / "validacao_execucoes.csv", index=False
    )

    for metrica, (titulo, casas, criterio) in METRICAS.items():
        if metrica not in resultados:
            continue
        tabela = formatar_tabela(resumo, metrica, casas, criterio)
        tabela_latex = formatar_tabela(
            resumo, metrica, casas, criterio, destacar=True
        )
        identificador = NOMES_ARQUIVOS.get(metrica, metrica)
        nome = f"tabela_{identificador}"
        tabela.to_csv(pasta_tabelas / f"{nome}.csv")
        (pasta_tabelas / f"{nome}.tex").write_text(
            converter_para_latex(
                tabela_latex,
                titulo=titulo,
                metrica=identificador,
                criterio=criterio,
            ),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
