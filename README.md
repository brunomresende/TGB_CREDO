# credal_cp_tgb

Implementacao didatica e reproduzivel do metodo **CREDO** (*Epistemic-Aware
Conformalized Credal Envelopes for Regression*).

O projeto separa explicitamente as quatro etapas do metodo:

```text
modelo quantilico -> amostras de quantis -> envelope credal -> calibracao conformal
```

## Objetivos

- reproduzir o protocolo experimental do artigo;
- manter CREDO com QNN/MC Dropout e BART heteroscedastico;
- comparar com CQR, CQR-r, UACQR-S, UACQR-P e EPICScore;
- incluir CQR-QNN e CQR-r-QNN para uma comparacao controlada;
- documentar as formulas em portugues e manter a API em ingles;
- registrar configuracoes, sementes e resultados brutos.

## Comparacao controlada com QNN

A mesma QNN treinada pode ser compartilhada pelos quatro metodos:

| Metodo | Dropout no treino | Dropout na inferencia |
|---|---:|---:|
| CQR-QNN | sim | nao |
| CQR-r-QNN | sim | nao |
| CREDO-QNN | sim | sim, MC Dropout |
| CREDO-QNN adaptativo | sim | sim, MC Dropout |

No MC Dropout, somente as camadas `Dropout` sao ativadas; `BatchNorm` permanece
em avaliacao para nao alterar suas estatisticas.

## Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,catboost]'
```

Para incluir BART:

```bash
pip install -e '.[full]'
```

## Testes

```bash
pytest
```

## Execucao

Modo rapido:

```bash
python scripts/run_experiment.py --config configs/benchmark_quick.yaml
```

O perfil completo preserva os valores do artigo, mas demanda muito tempo:

```bash
python scripts/run_all_benchmarks.py --config configs/benchmark_full.yaml
```

## Referencia

L. M. C. Cabezas et al., *CREDO: Epistemic-Aware Conformalized Credal
Envelopes for Regression*, arXiv:2603.06826.

