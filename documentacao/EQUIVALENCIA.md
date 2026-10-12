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
- dependências pesadas, como CatBoost e BART, são extras opcionais;
- nomes próprios de bibliotecas, siglas e argumentos externos não são traduzidos.

## Pontos a validar numericamente

- versão exata de CQR-r usada nas tabelas do artigo;
- estatísticas de ordem do UACQR-P para empates e limites infinitos;
- formato dos tensores posteriores do PyMC-BART nas versões fixadas;
- equivalência do EPICScore em todos os conjuntos;
- tolerâncias esperadas para resultados estocásticos.
