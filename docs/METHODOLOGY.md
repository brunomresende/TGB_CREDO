# Metodologia experimental

## Protocolo principal

- 56% treino, 24% calibracao e 20% teste;
- 30 repeticoes independentes no perfil completo;
- cobertura nominal de 90% (`alpha=0.1`);
- 1000 passagens MC Dropout para CREDO-QNN;
- 1000 estagios CatBoost para UACQR no perfil completo.

## Metodos na mesma tabela

1. CREDO-QNN;
2. CREDO-QNN adaptativo;
3. CQR-QNN;
4. CQR-r-QNN;
5. CQR-CatBoost;
6. CQR-r-CatBoost;
7. UACQR-S-CatBoost;
8. UACQR-P-CatBoost;
9. EPICScore-CatBoost-MDN.

A mesma QNN treinada e reutilizada nos quatro primeiros metodos. CQR utiliza
`eval()` (dropout desligado). CREDO ativa somente `Dropout` durante a inferencia,
preservando `BatchNorm` em avaliacao.

## Metricas

- AMC: cobertura marginal;
- ACO: cobertura nos outliers;
- SMIS: interval score medio;
- IL: comprimento medio;
- ILR: largura media nos outliers / largura nos 20% inliers mais centrais;
- PCorr: correlacao absoluta entre largura e indicador de cobertura.

Para CREDO tambem sao armazenadas as parcelas aleatoria, epistemica e conformal.

## Outliers

O diagnostico externo reproduz t-SNE em duas dimensoes seguido de LOF com 15
vizinhos e contaminacao de 5%. O mecanismo de gamma adaptativo nao acessa esses
rotulos. A decomposicao usa os 10% inliers mais centrais como referencia.

