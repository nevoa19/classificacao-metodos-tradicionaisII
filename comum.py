import time
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from dados import SEMENTE, subamostra
from modelos import GRADES, ALEATORIA, N_ITER, criar_modelo, criar_com_parametros
from metricas import calcular_metricas

def buscar_e_avaliar(nome, Xtr, ytr, Xte, yte, folds, sub_busca):
    r = {}

    Xs, ys = subamostra(Xtr, ytr, sub_busca)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEMENTE)
    if ALEATORIA[nome]:
        busca = RandomizedSearchCV(criar_modelo(nome, ys), GRADES[nome], n_iter=N_ITER[nome],
                                   scoring="f1", cv=cv, n_jobs=1, refit=False, random_state=SEMENTE)
    else:
        busca = GridSearchCV(criar_modelo(nome, ys), GRADES[nome], scoring="f1", cv=cv, n_jobs=-1, refit=False)

    t0 = time.time()
    busca.fit(Xs, ys)
    r["segundos_busca"] = round(time.time() - t0, 1)

    parametros = {}
    for chave in busca.best_params_:
        parametros[chave] = busca.best_params_[chave]
    r["parametros"] = parametros
    r["f1_vc"] = float(busca.best_score_)
    r["subamostra_busca"] = sub_busca
    if ALEATORIA[nome]:
        r["busca"] = "RandomizedSearchCV (n_iter=" + str(N_ITER[nome]) + ")"
    else:
        r["busca"] = "GridSearchCV"

    modelo = criar_com_parametros(nome, parametros, ytr)
    t0 = time.time()
    modelo.fit(Xtr, ytr)
    r["segundos_ajuste"] = round(time.time() - t0, 2)

    r["treino"] = calcular_metricas(ytr, modelo.predict(Xtr))
    prev_teste = modelo.predict(Xte)
    r["teste"] = calcular_metricas(yte, prev_teste)
    return r, modelo, prev_teste

def media(lista):
    soma = 0
    for v in lista:
        soma = soma + v
    return soma / len(lista)

def desvio(lista):
    m = media(lista)
    soma = 0
    for v in lista:
        soma = soma + (v - m) ** 2
    return (soma / len(lista)) ** 0.5

def fracao_diferentes(prev_a, prev_b):
    prev_a = list(prev_a)
    prev_b = list(prev_b)
    dif = 0
    i = 0
    while i < len(prev_a):
        if prev_a[i] != prev_b[i]:
            dif += 1
        i += 1
    return dif / len(prev_a)

def estabilidade(nome, parametros, Xtr, ytr, Xte, yte, n_rodadas, porc_removida):
    gerador = np.random.default_rng(SEMENTE)
    n = len(Xtr)
    n_manter = int(n * (1 - porc_removida))
    previsoes = []
    f1s = []
    raizes = []
    folhas = []
    b = 0
    while b < n_rodadas:
        idx = gerador.choice(n, n_manter, replace=False)
        Xb = Xtr.iloc[idx]
        yb = ytr.iloc[idx]
        modelo = criar_com_parametros(nome, parametros, yb)
        modelo.fit(Xb, yb)
        p = modelo.predict(Xte)
        previsoes.append(p)
        f1s.append(calcular_metricas(yte, p)["f1"])
        if nome == "arvore":
            arv = modelo.tree_
            raizes.append(str(Xtr.columns[arv.feature[0]]) + " <= " + ("%.2f" % arv.threshold[0]))
            folhas.append(int(modelo.get_n_leaves()))
        b += 1

    discord = []
    i = 0
    while i < n_rodadas:
        j = i + 1
        while j < n_rodadas:
            discord.append(fracao_diferentes(previsoes[i], previsoes[j]))
            j += 1
        i += 1

    e = {}
    e["rodadas"] = n_rodadas
    e["porc_removida"] = porc_removida
    e["f1_teste"] = f1s
    e["f1_media"] = media(f1s)
    e["f1_dp"] = desvio(f1s)
    e["discordancia_media"] = media(discord)
    if nome == "arvore":
        e["raizes"] = raizes
        e["folhas"] = folhas
    return e
