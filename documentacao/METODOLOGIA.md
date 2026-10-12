# Metodologia experimental

## Protocolo principal

- 56% para treino, 24% para calibração e 20% para teste;
- 30 repetições independentes no perfil completo;
- cobertura nominal de 90% (`alpha=0.1`);
- 1000 passagens de MC Dropout para CREDO-QNN;
- 1000 estágios do CatBoost para UACQR no perfil completo.

## Métodos na mesma tabela

1. CREDO-QNN;
2. CREDO-QNN adaptativo;
3. CQR-QNN;
4. CQR-r-QNN;
5. CQR-CatBoost;
6. CQR-r-CatBoost;
7. UACQR-S-CatBoost;
8. UACQR-P-CatBoost;
9. EPICScore-CatBoost-MDN.

A mesma QNN treinada é reutilizada nos quatro primeiros métodos. CQR utiliza
`eval()` com dropout desligado. CREDO ativa somente `Dropout` durante a
inferência, preservando `BatchNorm` em avaliação.

## Métricas

- AMC: cobertura marginal;
- ACO: cobertura nos valores atípicos;
- SMIS: escore médio do intervalo;
- IL: comprimento médio do intervalo;
- ILR: largura média nos valores atípicos dividida pela largura nos 20% dos
  pontos internos mais centrais;
- PCorr: correlação absoluta entre a largura e o indicador de cobertura.

Para CREDO também são armazenadas as parcelas aleatória, epistêmica e conformal.

## Valores atípicos

O diagnóstico externo reproduz t-SNE em duas dimensões seguido de LOF com 15
vizinhos e contaminação de 5%. O mecanismo de gamma adaptativo não acessa esses
rótulos. A decomposição usa os 10% dos pontos internos mais centrais como
referência.
