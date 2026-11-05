# credal_cp_tgb

Implementação didática e reproduzível do método **CREDO** (*Epistemic-Aware
Conformalized Credal Envelopes for Regression*).


O projeto separa explicitamente as quatro etapas do método:

```text
modelo quantílico -> amostras de quantis -> envelope credal -> calibração conformal
```

## Objetivos

- manter CREDO com QNN/MC Dropout e BART heteroscedástico;
- comparar com CQR, CQR-r, UACQR-S, UACQR-P e EPICScore;
- incluir CQR-QNN e CQR-r-QNN para uma comparação controlada;
- registrar configurações, sementes e resultados brutos.

## Estrutura do repositório

```text
configuracoes/       parâmetros das comparações
dados/               dados brutos e processados
experimentos/        exemplos didáticos numerados
resultados/           saídas brutas, resumos, tabelas e figuras
scripts/              pontos de entrada executáveis
src/credal_cp_tgb/    implementação da biblioteca
testes/               testes automatizados
```

### `configuracoes/`

Contém os arquivos YAML que definem os experimentos reproduzíveis. Neles ficam
valores como sementes, número de repetições, arquitetura da QNN e parâmetros de
gamma.

### `dados/`

Armazena os arquivos de dados, e não a lógica que os produz. `dados/brutos/`
recebe os arquivos originais baixados; `dados/processados/` recebe as versões
prontas para os modelos. O arquivo `dados/README.md` registra a origem e o formato
esperado dos conjuntos. Os dados sintéticos usados nas explicações são gerados por
código em `src/credal_cp_tgb/dados.py` e, portanto, não precisam ser salvos
aqui.


### `experimentos/`

Contém os programas didáticos que podem ser executados diretamente. Cada arquivo
conta uma história completa, com uma `principal()` curta indicando a ordem das
etapas. Esta é a melhor pasta para começar: leia primeiro
`01_introducao_cqr_e_credo.py` e depois passe às reproduções com BART e às avaliações mais demoradas.
Esses arquivos usam a biblioteca em `src/`, mas
mantêm visível a sequência dados → ajuste → calibração → previsão → figura.

### `resultados/`

É o destino das saídas produzidas pelos experimentos. `figuras/` guarda imagens,
`brutos/` preserva resultados de cada repetição, `resumos/` recebe agregações e
`tabelas/` guarda tabelas finais. Em geral, não é necessário ler esta pasta para
entender o método; ela existe para separar produtos gerados do código-fonte.

### `scripts/`

Contém pontos de entrada operacionais: baixar conjuntos, executar comparações e
agregar resultados. `executar_experimento.py` lê a linha de comando e chama
`comparacao.py`, que coordena os vários métodos. Esse código ficou aqui, fora de
`src/`, porque automatiza o estudo inteiro, mas não faz parte do algoritmo CREDO.
Para aprender o método, esta pasta pode ser ignorada inicialmente.

### `src/credal_cp_tgb/`

É a biblioteca propriamente dita. Os arquivos em `experimentos/` chamam as classes
e funções daqui. O núcleo conceitual está em `dados.py`, `modelos/`, `conformal/`,
`credal/` e `metodos/`; `avaliacao/` mede os resultados e `utilitarios/` contém
funções auxiliares pequenas. A automação de linha de comando foi mantida em
`scripts/`, deixando `src/` concentrado na implementação científica. Uma ordem
de leitura detalhada é apresentada mais abaixo.

### `testes/`

Contém testes pequenos e rápidos para verificar as fórmulas e os contratos do
projeto. Cada arquivo corresponde aproximadamente a um componente de `src/`, por
exemplo `teste_gamma.py`, `teste_envelope.py` e `teste_fluxo.py`. Além de detectar
erros, esses testes oferecem exemplos mínimos de entrada e saída e podem ajudar a
entender uma função isoladamente.

## Estrutura da biblioteca (`src/credal_cp_tgb`)

### `dados.py`

Responsável por criar dados sintéticos, dividir observações entre treino,
calibração e teste e carregar conjuntos processados. A classe
`DivisaoDados` apenas agrupa as seis matrizes/vetores resultantes para evitar que
eles sejam passados fora de ordem. Este é o primeiro módulo interno recomendado
para leitura.

### `modelos/`

Implementa os modelos que estimam quantis condicionais: QNN, BART, CatBoost e
floresta quantílica. `base.py` define somente os contratos comuns esperados pelos
métodos; os outros arquivos contêm implementações independentes porque usam
bibliotecas e algoritmos muito diferentes. Para a introdução, leia apenas
`rede_neural_quantilica.py`; BART e CatBoost podem ficar para depois.

### `conformal/`

Implementa a parte de predição conformal que é compartilhada por vários métodos.
`escores.py` calcula a não conformidade e o quantil empírico, enquanto
`calibracao.py` guarda o limiar encontrado e expande os limites inferiores e
superiores. Essa pasta não conhece QNN, BART ou CREDO: ela recebe apenas vetores
de respostas e limites intervalares.

### `credal/`

Contém as duas operações específicas que antecedem a conformalização no CREDO.
`gamma.py` determina quanto da posterior será aparado, de forma fixa ou adaptada
à escassez local; `envelope.py` agrega as amostras de quantis em um limite inferior
e um superior. Essa separação permite estudar e testar a construção credal sem
treinar um modelo completo.

### `metodos/`

Combina as peças anteriores em métodos completos com as operações `ajustar`,
`calibrar` e `prever`. `cqr.py` é o comparador conformal básico e `credo.py` é o
centro do projeto. `uacqr.py` e `epicscore.py` são comparadores avançados usados
na avaliação do artigo e podem ser ignorados numa primeira leitura.

### `avaliacao/`

Reúne o código que mede os intervalos depois da previsão. `metricas.py` calcula
cobertura, comprimento e escores; `valores_atipicos.py` identifica regiões úteis
para avaliar sensibilidade epistêmica; `agregacao.py` resume várias repetições.
Nada nesta pasta participa do treinamento ou altera os intervalos produzidos.

### `utilitarios/`

Guarda funções auxiliares que não pertencem a uma etapa estatística específica.
Atualmente contém a fixação coordenada das sementes aleatórias de NumPy e
PyTorch e o cálculo do tempo total de métodos que compartilham um modelo-base.
É uma pasta de apoio e não precisa ser estudada para compreender o método.

### Arquivos no nível principal da biblioteca

`dados.py` reúne as operações curtas de leitura, divisão e geração de dados que
antes estavam espalhadas numa pasta com vários arquivos. `__init__.py` define a
API pública curta, permitindo imports como
`from credal_cp_tgb import RegressorCREDO`. O orquestrador das comparações e a
linha de comando ficam agora em `scripts/`, pois são ferramentas de execução.

## Ordem recomendada de leitura

Para entender como os componentes se comunicam sem tentar ler todo o repositório,
siga esta sequência:

```text
1. experimentos/01_introducao_cqr_e_credo.py
                         ↓
2. src/credal_cp_tgb/dados.py
                         ↓
3. src/credal_cp_tgb/modelos/rede_neural_quantilica.py
                         ↓
4. src/credal_cp_tgb/conformal/escores.py e calibracao.py
                         ↓
5. src/credal_cp_tgb/credal/gamma.py e envelope.py
                         ↓
6. src/credal_cp_tgb/metodos/cqr.py e credo.py
```

Depois dessa sequência, `avaliacao/`, `scripts/comparacao.py`, UACQR e EPICScore ficam
mais fáceis de interpretar. Eles ampliam a avaliação, mas não são necessários
para compreender a ideia central do CREDO.

Os exemplos didáticos e simulados foram reunidos em oito histórias completas:

| Arquivo | Objetivo |
|---|---|
| `01_introducao_cqr_e_credo.py` | CQR, envelope credal, conformalização, gamma e decomposição |
| `02_credo_bart_vs_cqr_rfqr.py` | reprodução da figura histórica CREDO-BART versus CQR-RFQR |
| `03_esquema_tres_etapas_credo.py` | esquema original das três etapas com GPJax |
| `04_avaliacao_experimental.py` | comparações sintéticas, modelos-base e dados reais |
| `05_credo_bart_vs_bart.py` | comparação controlada entre CREDO-BART e CQR-BART |
| `06_figuras_cp_e_incertezas.py` | figuras conceituais de predição conformal e incerteza |
| `07_credo_gamma_adaptativo_vs_fixo.py` | comparação entre gamma adaptativo e fixo |
| `08_convergencia_quantis_verdadeiros.py` | convergência em relação aos quantis verdadeiros |

Os scripts `experimentos/credo_alpha_gamma_grid.py`,
`experimentos/credo_adaptive_gamma_grid.py` e
`experimentos/credo_adaptive_gamma_min_max_grid.py` geram as grades usadas para
analisar a sensibilidade dos intervalos aos parâmetros do CREDO.

Depois das execuções com dados reais,
`scripts/gerar_tabelas_resultados.py` consolida as métricas em tabelas e
`scripts/gerar_boxplot_decomposicao.py` compara a fração epistêmica em pontos
internos e atípicos.

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

Introdução didática:

```bash
uv run python experimentos/01_introducao_cqr_e_credo.py
```

As demais opções estão documentadas no início de cada arquivo. Para consultar
os cenários da avaliação experimental:

```bash
uv run python experimentos/04_avaliacao_experimental.py --help
```

### Comparações reproduzíveis

Modo rápido:

```bash
uv run python scripts/executar_experimento.py \
  --configuracao configuracoes/comparacao_rapida.yaml
```

O perfil completo preserva os valores experimentais do artigo, mas demanda muito
tempo:

```bash
uv run python scripts/executar_experimento.py \
  --configuracao configuracoes/comparacao_completa.yaml
```

## Referência

L. M. C. Cabezas et al., *CREDO: Epistemic-Aware Conformalized Credal
Envelopes for Regression*, arXiv:2603.06826.
