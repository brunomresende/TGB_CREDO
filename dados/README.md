# Dados

Os arquivos de dados brutos e processados não são versionados. Esta pasta mantém
apenas a estrutura esperada pelo projeto:

- `brutos/`: arquivos originais, sem alterações;
- `processados/`: um CSV por conjunto, pronto para os experimentos.

## Download dos conjuntos públicos

O script `scripts/baixar_conjuntos.py` baixa cinco conjuntos públicos diretamente
do UCI Machine Learning Repository:

| Nome no projeto | Conjunto original |
|---|---|
| `airfoil` | [Airfoil Self-Noise](https://archive.ics.uci.edu/ml/machine-learning-databases/00291/airfoil_self_noise.dat) |
| `concrete` | [Concrete Compressive Strength](https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls) |
| `protein` | [Physicochemical Properties of Protein Tertiary Structure](https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv) |
| `winered` | [Wine Quality — red](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) |
| `winewhite` | [Wine Quality — white](https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv) |

Execute o download a partir da raiz do repositório, informando um dos nomes da
primeira coluna. Por exemplo:

```bash
uv run python scripts/baixar_conjuntos.py airfoil
uv run python scripts/baixar_conjuntos.py concrete
```

Por padrão, cada arquivo é salvo em `dados/brutos/<nome>/`. Para escolher outro
caminho, use `--destino`:

```bash
uv run python scripts/baixar_conjuntos.py protein \
  --destino dados/brutos/protein/CASP.csv
```

O script apenas baixa o arquivo original: ele não transforma os dados nem gera o
CSV processado.

## Conjuntos de obtenção manual

Os demais conjuntos da comparação completa não são baixados automaticamente:
`star`, `cycle`, `electric`, `meps19`, `superconductivity`, `homes` e `WEC`.
Eles devem ser obtidos de suas fontes originais, respeitando as respectivas
licenças e eventuais termos de acesso, e então preparados no formato descrito
abaixo.

## Formato processado

Cada conjunto deve ser salvo diretamente em `dados/processados/`, em formato CSV,
somente com colunas numéricas. As variáveis explicativas podem conservar seus
nomes, mas a variável resposta deve se chamar obrigatoriamente `alvo`.

Exemplo mínimo:

```csv
variavel_1,variavel_2,alvo
0.25,1.30,4.10
0.80,0.75,3.60
```

A comparação completa, definida em
`configuracoes/comparacao_completa.yaml`, espera estes arquivos:

```text
dados/processados/concrete.csv
dados/processados/airfoil.csv
dados/processados/winewhite.csv
dados/processados/star.csv
dados/processados/winered.csv
dados/processados/cycle.csv
dados/processados/electric.csv
dados/processados/meps19.csv
dados/processados/superconductivity.csv
dados/processados/homes.csv
dados/processados/protein.csv
dados/processados/WEC.csv
```

Os nomes diferenciam maiúsculas de minúsculas em sistemas como Linux; por isso,
`WEC.csv` deve conservar as letras maiúsculas. Os comandos de download não
sobrescrevem nem alteram arquivos já existentes em `dados/processados/`.
