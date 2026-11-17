import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree
from modelos import MODELOS, NOMES, NOMES_MINI1
from tabelas import numero_br

PASTA_FIG = "resultados/figuras"

AZUL = '#0275d8'
VERMELHO = '#d9534f'
CINZA = '#777777'

def salvar_figura(nome):
    if not os.path.exists(PASTA_FIG):
        os.makedirs(PASTA_FIG)
    plt.savefig(PASTA_FIG + "/" + nome + ".pdf", bbox_inches="tight")
    plt.savefig(PASTA_FIG + "/" + nome + ".png", dpi=200, bbox_inches="tight")
    plt.close()

def grafico_matrizes(resultados, tag, classes):
    fig, eixos = plt.subplots(1, len(MODELOS), figsize=(1.9 * len(MODELOS), 2.0), dpi=150, sharey=True)
    fig.subplots_adjust(wspace=0.15)
    j = 0
    while j < len(MODELOS):
        m = MODELOS[j]
        cm = resultados[m]["teste"]["confusao"]
        norm = []
        a = 0
        while a < 2:
            total_linha = cm[a][0] + cm[a][1]
            norm.append([cm[a][0] / total_linha, cm[a][1] / total_linha])
            a += 1
        eixos[j].imshow(norm, cmap="Blues", vmin=0, vmax=1)
        a = 0
        while a < 2:
            b = 0
            while b < 2:
                cor = "black"
                if norm[a][b] > 0.5:
                    cor = "white"
                eixos[j].text(b, a, numero_br(cm[a][b]), ha="center", va="center", color=cor, fontsize=7)
                b += 1
            a += 1
        eixos[j].set_xticks([0, 1], classes, fontsize=6)
        if j == 0:
            eixos[j].set_yticks([0, 1], classes, fontsize=6)
            eixos[j].set_ylabel("real", fontsize=7)
        eixos[j].set_title(NOMES[m], fontsize=9)
        eixos[j].set_xlabel("previsto", fontsize=7)
        j += 1
    salvar_figura(tag + "_matrizes_confusao")

def grafico_arvore(modelo, colunas, tag, classes):
    plt.figure(figsize=(7.5, 3.6), dpi=150)
    plot_tree(modelo, max_depth=1, feature_names=list(colunas), class_names=classes,
              filled=True, rounded=True, impurity=True, proportion=True, fontsize=9)
    salvar_figura(tag + "_arvore")

def grafico_profundidade(curva, melhor_prof, tag):
    plt.figure(figsize=(3.4, 2.4), dpi=150)
    plt.plot(curva["profundidade"], curva["treino"], "o-", ms=3, color=AZUL, label="treino")
    plt.plot(curva["profundidade"], curva["teste"], "s-", ms=3, color=VERMELHO, label="teste")
    if melhor_prof is not None:
        plt.axvline(melhor_prof, color=CINZA, ls="--", lw=0.8, label="escolhida (VC)")
    plt.xlabel("max_depth", fontsize=8)
    plt.ylabel("F1", fontsize=8)
    plt.ylim(0, 1.02)
    plt.tick_params(labelsize=7)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False, fontsize=7)
    salvar_figura(tag + "_profundidade")

def grafico_n_arvores(curva, tag):
    plt.figure(figsize=(3.4, 2.4), dpi=150)
    plt.plot(curva["n"], curva["treino_f1"], "o-", ms=3, color=AZUL, label="F1 treino")
    plt.plot(curva["n"], curva["teste_f1"], "s-", ms=3, color=VERMELHO, label="F1 teste")
    plt.xscale("log")
    plt.xlabel("nº de árvores (n_estimators, escala log)", fontsize=8)
    plt.ylabel("F1", fontsize=8)
    plt.tick_params(labelsize=7)
    plt.grid(True, ls="--", alpha=0.4)
    plt.legend(frameon=False, fontsize=7)
    salvar_figura(tag + "_n_arvores")

def grafico_importancias(modelos_treinados, colunas, tag, n_top):
    fig, eixos = plt.subplots(len(MODELOS), 1, figsize=(3.6, 2.0 * len(MODELOS)), dpi=150)
    fig.subplots_adjust(hspace=0.45)
    j = 0
    while j < len(MODELOS):
        m = MODELOS[j]
        imp = modelos_treinados[m].feature_importances_
        pares = []
        k = 0
        while k < len(imp):
            pares.append([float(imp[k]), colunas[k]])
            k += 1
        pares.sort(reverse=True)
        nomes = []
        valores = []
        k = 0
        while k < n_top and k < len(pares):
            nomes.append(pares[k][1])
            valores.append(pares[k][0])
            k += 1
        nomes.reverse()
        valores.reverse()
        eixos[j].barh(nomes, valores, color=AZUL)
        eixos[j].set_title(NOMES[m], fontsize=9)
        eixos[j].tick_params(labelsize=6)
        j += 1
    salvar_figura(tag + "_importancias")

def grafico_painel(todos, mini1):
    tags = list(todos.keys())
    fig, eixos = plt.subplots(1, len(tags), figsize=(3.6 * len(tags), 2.6), dpi=150, squeeze=False)
    t = 0
    while t < len(tags):
        tag = tags[t]
        ax = eixos[0][t]
        nomes = []
        valores = []
        cores = []
        if mini1 is not None and tag in mini1:
            for m in NOMES_MINI1:
                if m in mini1[tag]["modelos"]:
                    nomes.append(NOMES_MINI1[m])
                    valores.append(mini1[tag]["modelos"][m]["teste"]["f1"])
                    cores.append(CINZA)
        for m in MODELOS:
            nomes.append(NOMES[m])
            valores.append(todos[tag]["modelos"][m]["teste"]["f1"])
            cores.append(AZUL)
        posicoes = []
        i = 0
        while i < len(nomes):
            posicoes.append(i)
            i += 1
        ax.bar(posicoes, valores, color=cores)
        i = 0
        while i < len(valores):
            ax.text(i, valores[i] + 0.01, numero_br(valores[i], 2), ha="center", fontsize=6)
            i += 1
        ax.set_xticks(posicoes, nomes, fontsize=6, rotation=35, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("F1 (teste)", fontsize=8)
        ax.set_title(todos[tag]["titulo"], fontsize=9, fontweight='bold')
        ax.grid(True, axis="y", ls="--", alpha=0.5)
        t += 1
    salvar_figura("painel_comparativo")
