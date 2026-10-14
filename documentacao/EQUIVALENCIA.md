# Política de equivalência

O objetivo inicial é reproduzir, e não corrigir silenciosamente, a implementação
de referência. Mudanças deliberadas devem receber teste e justificativa neste
documento.

## Preservado da QNN original

- camadas 128-128-64 no perfil completo;
- `Linear -> BatchNorm -> ReLU -> Dropout`;
- duas saídas quantílicas livres;
- soma das perdas pinball dos dois quantis;
- Adam, corte de gradiente em 1, StepLR e parada antecipada;
- padronização de X e y;
- correção de cruzamento por ordenação em cada passagem;
- ativação exclusiva de `Dropout` no MC Dropout.

## Decisões tornadas explícitas

- o quantil conformal é uma estatística de ordem, sem interpolação;
- CQR-QNN e CREDO-QNN compartilham os mesmos pesos;
- resultados brutos são salvos antes da agregação;
- média, desvio-padrão, erro-padrão e IC95 são colunas distintas;
- as sementes das repetições usam ``RandomState(seed).randint``, reproduzindo a
  sequência legada do NumPy empregada no código oficial;
- a coluna exibida entre parênteses nas tabelas oficiais é ``2 × erro-padrão``
  (apesar de a legenda do artigo chamá-la de ``2 × desvio-padrão``);
- dependências pesadas, como CatBoost e BART, são extras opcionais;
- nomes próprios de bibliotecas, siglas e argumentos externos não são traduzidos.
- o gamma adaptativo usa por padrão `k = ceil(d log(n))`, como no código
  oficial; a heurística de potência permanece disponível apenas quando pedida.

## Figura sintética histórica CREDO-BART versus CQR-RFQR

O experimento `02_credo_bart_vs_cqr_rfqr.py` corresponde a
`credo_diferent_data_Figure1.py` do repositório original. Foram preservados:

- o processo gerador heteroscedástico;
- a região de escassez `[0, 0.4]`, com 2% das observações;
- 1500 observações e divisão 50%/25%/25%;
- CREDO com BART, gamma adaptativo e decomposição aleatória/epistêmica;
- o arranjo visual em quatro painéis, com o último vazio.

A figura histórica usa RFQR como modelo-base do CQR. CatBoost permanece no
protocolo das tabelas, mas não deve substituir o RFQR nesta ilustração. Foram
mantidas as sementes do script original: 125 para o estado global, 42 para
dados e divisão, 123 para o CQR e 1250 para o ajuste BART. O gerador local
baseado em `RandomState` preserva a sequência pseudoaleatória antiga.

## Pontos a validar numericamente

- versão exata de CQR-r usada nas tabelas do artigo;
- estatísticas de ordem do UACQR-P para empates e limites infinitos;
- formato dos tensores posteriores do PyMC-BART nas versões fixadas;
- equivalência do EPICScore em todos os conjuntos;
- tolerâncias esperadas para resultados estocásticos.
