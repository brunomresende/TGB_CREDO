# credal_cp_tgb

Implementação didática e reproduzível do método **CREDO** (*Epistemic-Aware
Conformalized Credal Envelopes for Regression*).

O código, a documentação, os nomes de funções, as classes, os experimentos e as
configurações estão em português. Permanecem em inglês apenas nomes impostos por
bibliotecas externas, siglas científicas e convenções da linguagem, como o método
`forward` do PyTorch e o diretório de empacotamento `src`.

O projeto separa explicitamente as quatro etapas do método:

```text
modelo quantílico -> amostras de quantis -> envelope credal -> calibração conformal
```

## Objetivos

- reproduzir o protocolo experimental do artigo;
- manter CREDO com QNN/MC Dropout e BART heteroscedástico;
- comparar com CQR, CQR-r, UACQR-S, UACQR-P e EPICScore;
- incluir CQR-QNN e CQR-r-QNN para uma comparação controlada;
- explicar as fórmulas e a implementação em português;
- registrar configurações, sementes e resultados brutos.

## Estrutura

```text
configuracoes/       parâmetros das comparações
dados/               dados brutos e processados
documentacao/        metodologia e decisões de equivalência
experimentos/        exemplos didáticos numerados
resultados/           saídas brutas, resumos, tabelas e figuras
scripts/              pontos de entrada executáveis
src/credal_cp_tgb/    implementação da biblioteca
testes/               testes automatizados
```

Dentro da biblioteca, os módulos principais são `modelos`, `metodos`, `conformal`,
`credal`, `avaliacao`, `dados` e `utilitarios`.

## Comparação controlada com QNN

A mesma QNN treinada é compartilhada pelos quatro métodos:

| Método | Dropout no treino | Dropout na inferência |
|---|---:|---:|
| CQR-QNN | sim | não |
| CQR-r-QNN | sim | não |
| CREDO-QNN | sim | sim, MC Dropout |
| CREDO-QNN adaptativo | sim | sim, MC Dropout |

No MC Dropout, somente as camadas `Dropout` são ativadas. `BatchNorm` permanece
em avaliação para não alterar suas estatísticas.

## Instalação

Com `uv`, incluindo todos os métodos e ferramentas de desenvolvimento:

```bash
uv sync --extra completo
```

Com `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[completo]'
```

## Testes

```bash
uv run pytest
```

## Execução

Modo rápido:

```bash
uv run python scripts/executar_experimento.py \
  --configuracao configuracoes/comparacao_rapida.yaml
```

O perfil completo preserva os valores experimentais do artigo, mas demanda muito
tempo:

```bash
uv run python scripts/executar_todas_comparacoes.py \
  --configuracao configuracoes/comparacao_completa.yaml
```

## Referência

L. M. C. Cabezas et al., *CREDO: Epistemic-Aware Conformalized Credal
Envelopes for Regression*, arXiv:2603.06826.
