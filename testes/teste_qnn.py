import numpy as np
import torch

from credal_cp_tgb.modelos.rede_neural_quantilica import RedeNeuralQuantilica


def teste_semente_tambem_controla_inicializacao_dos_pesos():
    primeiro = RedeNeuralQuantilica(2, camadas_ocultas=(8,), semente_aleatoria=7)
    _ = torch.rand(20)
    segundo = RedeNeuralQuantilica(2, camadas_ocultas=(8,), semente_aleatoria=7)

    for peso_primeiro, peso_segundo in zip(
        primeiro.rede.parameters(), segundo.rede.parameters(), strict=True
    ):
        torch.testing.assert_close(peso_primeiro, peso_segundo)


def teste_predicao_deterministica_e_modos_dropout_monte_carlo():
    gerador = np.random.default_rng(2)
    X = gerador.normal(size=(80, 2)).astype(np.float32)
    y = (X[:, 0] + gerador.normal(0, 0.1, 80)).astype(np.float32)
    modelo = RedeNeuralQuantilica(
        2,
        camadas_ocultas=(8,),
        epocas=3,
        paciencia=2,
        tamanho_lote=16,
        semente_aleatoria=2,
    ).ajustar(X, y)
    primeira = modelo.prever_quantis(X[:5])
    segunda = modelo.prever_quantis(X[:5])
    np.testing.assert_allclose(primeira, segunda)

    normalizacao_lote = next(
        modulo for modulo in modelo.rede.modules() if modulo.__class__.__name__ == "BatchNorm1d"
    )
    media_corrente = normalizacao_lote.running_mean.detach().clone()
    inferior, superior = modelo.amostrar_quantis(
        X[:5],
        numero_amostras=5,
        semente_aleatoria=3,
    )
    assert inferior.shape == superior.shape == (5, 5)
    assert np.any(np.std(inferior, axis=0) > 0)
    np.testing.assert_allclose(
        normalizacao_lote.running_mean.detach().numpy(),
        media_corrente.numpy(),
    )
