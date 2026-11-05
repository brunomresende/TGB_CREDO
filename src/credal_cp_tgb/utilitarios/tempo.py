"""Utilitários para cronometrar procedimentos com modelos compartilhados."""

from __future__ import annotations


def calcular_tempo_total_metodo(
    segundos_ajuste_compartilhado: float,
    inicio_etapas_especificas: float,
    fim_etapas_especificas: float,
) -> float:
    """Soma o treino compartilhado às etapas exclusivas do método."""
    return segundos_ajuste_compartilhado + (
        fim_etapas_especificas - inicio_etapas_especificas
    )
