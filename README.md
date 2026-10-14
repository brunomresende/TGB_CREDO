# credal_cp_tgb

## Resumo

Implementação em português do método **CREDO** (*Epistemic-Aware Conformalized Credal Envelopes for Regression*). O projeto contém exemplos didáticos, reprodução dos experimentos do artigo e comparações com CQR, CQR-r, UACQR-S, UACQR-P e EPICScore. Também inclui CQR-QNN e CQR-r-QNN como métodos adicionais para comparações com a mesma rede quantílica usada pelo CREDO.

## Ideia geral

O CREDO começa com um modelo que estima quantis condicionais e gera várias previsões posteriores. Essas previsões formam um envelope credal que representa a incerteza epistêmica. Por fim, a calibração conformal expande o envelope para obter cobertura marginal sem fazer suposições sobre a distribuição dos dados.

```text
dados
  ↓
modelo quantílico
  ↓
amostras posteriores de quantis
  ↓
envelope credal
  ↓
calibração conformal
  ↓
intervalo preditivo
```

O comprimento anterior à conformalização é decomposto em uma parcela aleatória, associada ao intervalo quantílico central, e uma parcela epistêmica, associada à expansão do envelope. A correção conformal é registrada separadamente.

## Diretórios

### `configuracoes/`

Contém arquivos YAML com sementes, conjuntos de dados, número de repetições, arquiteturas e hiperparâmetros usados nas execuções.

### `dados/`

Contém os arquivos originais em `brutos/` e os CSVs prontos para uso em `processados/`. Cada CSV processado deve ter a variável resposta na coluna `alvo`.

### `documentacao/`

Contém a descrição do protocolo em `METODOLOGIA.md` e o registro das correspondências com o código original em `EQUIVALENCIA.md`.

### `experimentos/`

Contém exemplos executáveis que apresentam os métodos, reproduzem figuras e realizam avaliações sintéticas ou com dados reais.

### `resultados/`

Contém métricas de cada repetição, resumos agregados, configurações utilizadas, sementes, informações do ambiente, tabelas e figuras produzidas nas execuções.

### `scripts/`

Contém os comandos para baixar dados, executar o protocolo experimental, coordenar os métodos e agregar resultados.

### `src/credal_cp_tgb/`

Contém a implementação dos dados, modelos quantílicos, calibração conformal, envelopes credais, métodos completos, métricas e utilitários.

### `testes/`

Contém testes automatizados para os geradores de dados, divisões, sementes, escores, envelopes, gamma, modelos e fluxos completos.

## Experimentos

| Arquivo | O que faz |
|---|---|
| `01_introducao_cqr_e_credo.py` | Apresenta CQR, CREDO, gamma e decomposição de incerteza com um exemplo pequeno. |
| `02_credo_bart_vs_cqr_rfqr.py` | Reproduz a figura sintética que compara CREDO-BART com CQR usando floresta quantílica. |
| `03_esquema_tres_etapas_credo.py` | Reproduz o esquema de três etapas do artigo usando um Processo Gaussiano implementado com GPJax. |
| `04_avaliacao_experimental.py` | Executa comparações sintéticas, compara modelos-base ou chama a avaliação com dados reais. |
| `05_credo_bart_vs_bart.py` | Compara CREDO-BART e CQR-BART usando o mesmo tipo de modelo-base. |

## Scripts

| Arquivo | O que faz |
|---|---|
| `baixar_conjuntos.py` | Baixa os conjuntos públicos cadastrados para `dados/brutos/`. |
| `executar_experimento.py` | Lê os argumentos do terminal e inicia uma comparação definida por um arquivo YAML. |
| `comparacao.py` | Treina os métodos, realiza as repetições, calcula métricas e salva os resultados. |
| `agregar_resultados.py` | Combina arquivos de métricas e produz um resumo agregado. |

## Configurações

| Arquivo | O que faz |
|---|---|
| `comparacao_rapida.yaml` | Executa um teste curto do protocolo usando Airfoil. |
| `comparacao_completa.yaml` | Define o protocolo completo para todos os conjuntos do artigo. |
| `airfoil_artigo.yaml` | Define o protocolo completo exclusivamente para Airfoil. |
| `sintetico.yaml` | Define parâmetros para experimentos com dados sintéticos. |

## Biblioteca

### `dados.py`

Gera os dados sintéticos, lê CSVs processados e divide as observações em treino, calibração e teste.

### `modelos/`

Contém a QNN com MC Dropout, BART heteroscedástico, CatBoost quantílico, floresta quantílica e a interface comum dos modelos-base.

### `conformal/`

Calcula escores de não conformidade, quantis empíricos e a expansão conformal dos limites inferior e superior.

### `credal/`

Calcula gamma fixo ou adaptativo e transforma amostras posteriores de quantis em um envelope credal.

### `metodos/`

Implementa os fluxos completos de CREDO, CQR, CQR-r, UACQR-S, UACQR-P e EPICScore com as operações de ajuste, calibração e previsão.

### `avaliacao/`

Calcula cobertura, SMIS, comprimento, ILR, cobertura em outliers, correlação cobertura–largura e estatísticas agregadas das repetições.

### `utilitarios/`

Contém a fixação das fontes de aleatoriedade e a geração das sementes usadas nas repetições do artigo.

### `__init__.py`

Expõe a API principal do pacote: `RegressorCREDO`, `RegressorCQR`, `GammaFixo` e `GammaAdaptativo`.

## Instalação

Com `uv`:

```bash
uv sync --extra completo
```

Com `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[completo]'
```

## Execução

Exemplo introdutório:

```bash
uv run python experimentos/01_introducao_cqr_e_credo.py
```

Teste rápido com Airfoil:

```bash
uv run python scripts/executar_experimento.py \
  --configuracao configuracoes/comparacao_rapida.yaml
```

Protocolo completo somente com Airfoil:

```bash
uv run python scripts/executar_experimento.py \
  --configuracao configuracoes/airfoil_artigo.yaml
```

Protocolo completo com todos os conjuntos:

```bash
uv run python scripts/executar_experimento.py \
  --configuracao configuracoes/comparacao_completa.yaml
```

Testes:

```bash
uv run pytest
```

## Referência

L. M. C. Cabezas et al., *CREDO: Epistemic-Aware Conformalized Credal Envelopes for Regression*, arXiv:2603.06826.
