# Politica de equivalencia

O objetivo inicial e reproduzir, e nao corrigir silenciosamente, a implementacao
de referencia. Mudancas deliberadas devem receber teste e justificativa neste
documento.

## Preservado da QNN original

- camadas 128-128-64 no perfil completo;
- `Linear -> BatchNorm -> ReLU -> Dropout`;
- duas saidas quantilicas livres;
- soma das pinball losses dos dois quantis;
- Adam, clipping de gradiente em 1, StepLR e early stopping;
- padronizacao de X e y;
- correcao de crossing por ordenacao em cada passagem;
- ativacao exclusiva de Dropout no MC Dropout.

## Decisoes tornadas explicitas

- o quantil conformal e uma estatistica de ordem, sem interpolacao;
- CQR-QNN e CREDO-QNN compartilham os mesmos pesos;
- resultados brutos sao salvos antes da agregacao;
- media, desvio-padrao, erro-padrao e IC95 sao colunas distintas;
- dependencias pesadas (CatBoost/BART) sao extras opcionais.

## Pontos a validar numericamente

- versao exata de CQR-r usada nas tabelas do artigo;
- estatisticas de ordem UACQR-P para empates e limites infinitos;
- formato dos tensores posteriores do PyMC-BART nas versoes fixadas;
- equivalencia do EPICScore em todos os datasets;
- tolerancias esperadas para resultados estocasticos.

