"""Reproduz o esquema original das três etapas do CREDO.

Somente nomes, títulos e legendas foram traduzidos. Simulação, GPJax,
sementes, kernel, amostragens e quantis são os do script oficial.

Execute:
    uv run --extra figura-esquema python experimentos/03_esquema_tres_etapas_credo.py
"""

from pathlib import Path

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax import config
from jax.scipy.stats import norm
from jaxtyping import install_import_hook
from matplotlib.lines import Line2D

config.update("jax_enable_x64", True)
with install_import_hook("gpjax", "beartype.beartype"):
    import gpjax as gpx

RAIZ_PROJETO = Path(__file__).resolve().parents[1]
ARQUIVO_SAIDA = RAIZ_PROJETO / "resultados" / "figuras" / "esquema_tres_etapas_credo.png"

plt.rcParams.update({"font.size": 16, "axes.labelsize": 16, "xtick.labelsize": 14,
                     "ytick.labelsize": 14, "legend.fontsize": 14, "axes.titlesize": 16})


def simular_dados(semente=7, numero_grade=220, numero_pontos=200):
    gerador = np.random.default_rng(semente)
    grade_x = np.linspace(0, 1, numero_grade)
    media = (0.25 + 0.55 * np.sin(2 * np.pi * (grade_x - 0.08))
             - 0.25 * np.cos(2 * np.pi * (grade_x - 0.15)) + 1.15 * grade_x**3)
    x = gerador.uniform(0.08, 0.85, size=numero_pontos)
    media_x = np.interp(x, grade_x, media)
    y = media_x + gerador.normal(0, 0.18 + 0.10 * (1 - x), size=numero_pontos)
    return grade_x, x, y


def calcular_curvas():
    grade_x, x, y = simular_dados()
    chave = jr.key(123)
    dados = gpx.Dataset(X=jnp.array(x.reshape(-1, 1), dtype=jnp.float64),
                        y=jnp.array(y.reshape(-1, 1), dtype=jnp.float64))
    kernel = gpx.kernels.RBF() + (gpx.kernels.Matern52(lengthscale=0.12) * gpx.kernels.RBF())
    prior = gpx.gps.Prior(mean_function=gpx.mean_functions.Zero(), kernel=kernel)
    verossimilhanca = gpx.likelihoods.Gaussian(num_datapoints=dados.n)
    posterior = prior * verossimilhanca
    posterior_otimizado, _ = gpx.fit_scipy(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=dados,
        trainable=gpx.parameters.Parameter,
    )
    print(-gpx.objectives.conjugate_mll(posterior_otimizado, dados))

    chave, subchave = jr.split(chave)
    distribuicao = posterior.predict(grade_x.reshape(-1, 1), train_data=dados)
    funcoes = distribuicao.sample(subchave, sample_shape=(30,))
    sigma = posterior_otimizado.likelihood.obs_stddev.get_value()
    escores_z = norm.ppf(jnp.array([0.05, 0.95]))
    funcoes_quantis = funcoes[..., None] + escores_z * sigma
    inferiores, superiores = funcoes_quantis[:, :, 0], funcoes_quantis[:, :, 1]
    envelope_inferior = np.quantile(inferiores, 0.1 / 2, axis=0)
    envelope_superior = np.quantile(superiores, 1 - 0.1 / 2, axis=0)

    _, x_calibracao, y_calibracao = simular_dados(semente=125)
    X_calibracao = jnp.array(x_calibracao.reshape(-1, 1), dtype=jnp.float64)
    y_calibracao = jnp.array(y_calibracao.reshape(-1, 1), dtype=jnp.float64)
    chave, subchave = jr.split(chave)
    distribuicao_cal = posterior.predict(X_calibracao, train_data=dados)
    funcoes_cal = distribuicao_cal.sample(subchave, sample_shape=(1000,))
    quantis_cal = funcoes_cal[..., None] + escores_z * sigma
    inferior_cal = np.quantile(quantis_cal[:, :, 0], 0.1 / 2, axis=0)
    superior_cal = np.quantile(quantis_cal[:, :, 1], 1 - 0.1 / 2, axis=0)
    tamanho = y_calibracao.shape[0]
    escores = jnp.maximum(inferior_cal - y_calibracao.flatten(),
                          y_calibracao.flatten() - superior_cal)
    corte = np.quantile(escores, np.ceil((tamanho + 1) * (1 - 0.1)) / tamanho)
    return (grade_x, x, y, inferiores, superiores, envelope_inferior,
            envelope_superior, envelope_inferior - corte, envelope_superior + corte)


def desenhar():
    (grade_x, x, y, inferiores, superiores, envelope_inferior, envelope_superior,
     conformal_inferior, conformal_superior) = calcular_curvas()
    cor_inferior, cor_superior = plt.rcParams["axes.prop_cycle"].by_key()["color"][:2]
    figura = plt.figure(figsize=(11, 7.2))
    layout = figura.add_gridspec(2, 4, hspace=0.58, wspace=0.45)
    eixo_1 = figura.add_subplot(layout[0, 0:2])
    eixo_2 = figura.add_subplot(layout[0, 2:4])
    eixo_3 = figura.add_subplot(layout[1, 1:3])

    for indice in range(inferiores.shape[0]):
        for eixo in (eixo_1, eixo_2):
            eixo.plot(grade_x, inferiores[indice], color=cor_inferior, alpha=0.16, linewidth=1)
            eixo.plot(grade_x, superiores[indice], color=cor_superior, alpha=0.16, linewidth=1)

    eixo_1.scatter(x, y, s=18, color=cor_inferior, alpha=0.75, zorder=3)
    eixo_1.set_title(r"$\mathcal{F}=\{F_\theta(\cdot\mid x):\theta\in\Theta\}$")
    eixo_1.legend(handles=[
        Line2D([0], [0], color=cor_inferior, linewidth=3.2, alpha=0.5,
               label=r"$F_\theta^{-1}(\alpha/2\mid x)$"),
        Line2D([0], [0], color=cor_superior, linewidth=3.2, alpha=0.5,
               label=r"$F_\theta^{-1}(1-\alpha/2\mid x)$")], loc="upper left")

    eixo_2.scatter(x, y, s=18, color=cor_inferior, alpha=0.45, zorder=3)
    eixo_2.plot(grade_x, envelope_inferior, color=cor_inferior, alpha=0.98,
                linewidth=3.2, label=r"$\ell(x)$")
    eixo_2.plot(grade_x, envelope_superior, color=cor_superior, alpha=0.98,
                linewidth=3.2, label=r"$u(x)$")
    eixo_2.fill_between(grade_x, envelope_inferior, envelope_superior,
                        color="lightgray", alpha=0.45, zorder=2)
    eixo_2.set_title(r"$\mathcal{F}_{qq}(x)=\{F_\theta(\cdot\mid x):\theta\in R_{qq}(x)\}$")
    eixo_2.legend(loc="upper left")

    eixo_3.scatter(x, y, s=18, color=cor_inferior, alpha=0.45, zorder=3)
    eixo_3.plot(grade_x, envelope_inferior, color=cor_inferior, alpha=0.5, linewidth=3.2)
    eixo_3.plot(grade_x, envelope_superior, color=cor_superior, alpha=0.5, linewidth=3.2)
    eixo_3.plot(grade_x, conformal_inferior, color=cor_inferior, alpha=0.95,
                linewidth=3.2, label=r"$\ell(x)-\hat\tau$")
    eixo_3.plot(grade_x, conformal_superior, color=cor_superior, alpha=0.95,
                linewidth=3.2, label=r"$u(x)+\hat\tau$")
    eixo_3.fill_between(grade_x, conformal_inferior, conformal_superior,
                        color="lightgray", alpha=0.45, zorder=2)
    eixo_3.set_title(r"Conformalização de $[\ell(x),u(x)]$")
    eixo_3.legend(loc="upper left")

    for eixo in (eixo_1, eixo_2, eixo_3):
        eixo.set(xlim=(0, 1), xlabel="x", ylabel="y")
    eixo_1.text(0.5, 1.23, "Etapa 1: modelo aleatório", color="darkred", fontsize=16,
                ha="center", transform=eixo_1.transAxes)
    eixo_2.text(0.5, 1.23, "Etapa 2: envelope credal aparado", color="royalblue",
                fontsize=16, ha="center", transform=eixo_2.transAxes)
    eixo_3.text(0.5, 1.23, "Etapa 3: conformalização", color="forestgreen", fontsize=16,
                ha="center", transform=eixo_3.transAxes)
    eixo_1.annotate("", xy=(1.13, 0.5), xytext=(1.01, 0.5), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "simple", "color": "black"})
    eixo_2.annotate("", xy=(0.55, -0.35), xytext=(0.7, -0.02), xycoords="axes fraction",
                    arrowprops={"arrowstyle": "simple", "color": "black"})
    ARQUIVO_SAIDA.parent.mkdir(parents=True, exist_ok=True)
    figura.savefig(ARQUIVO_SAIDA, dpi=300, bbox_inches="tight", transparent=True)
    plt.close(figura)
    print(f"Figura salva em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    desenhar()
