import time
from modelos import criar_com_parametros
from metricas import calcular_metricas
from comum import buscar_e_avaliar, estabilidade

LISTA_N_ARVORES = [1, 2, 5, 10, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500]

def rodar_floresta(Xtr, ytr, Xte, yte, folds, sub_busca, rodadas_estab):
    r, modelo, prev = buscar_e_avaliar("floresta", Xtr, ytr, Xte, yte, folds, sub_busca)
    r["curva"] = curva_n_arvores(r["parametros"], Xtr, ytr, Xte, yte)
    r["estabilidade"] = estabilidade("floresta", r["parametros"], Xtr, ytr, Xte, yte, rodadas_estab, 0.05)
    return r, modelo, prev

def curva_n_arvores(parametros, Xtr, ytr, Xte, yte):
    params = {}
    for k in parametros:
        if k != "n_estimators":
            params[k] = parametros[k]

    qtd = [0, 0]
    for v in ytr:
        qtd[int(v)] += 1
    pesos = {}
    pesos[0] = len(ytr) / (2 * qtd[0])
    pesos[1] = len(ytr) / (2 * qtd[1])
    params["class_weight"] = pesos

    modelo = criar_com_parametros("floresta", params)
    modelo.warm_start = True

    curva = {}
    curva["n"] = []
    curva["treino_f1"] = []
    curva["teste_f1"] = []
    curva["teste_acuracia"] = []
    curva["segundos_acumulados"] = []
    t0 = time.time()
    for n in LISTA_N_ARVORES:
        modelo.n_estimators = n
        modelo.fit(Xtr, ytr)
        seg = time.time() - t0
        m_tr = calcular_metricas(ytr, modelo.predict(Xtr))
        m_te = calcular_metricas(yte, modelo.predict(Xte))
        curva["n"].append(n)
        curva["treino_f1"].append(m_tr["f1"])
        curva["teste_f1"].append(m_te["f1"])
        curva["teste_acuracia"].append(m_te["acuracia"])
        curva["segundos_acumulados"].append(round(seg, 2))
        print(f"   floresta n={n:4d}  F1 treino={m_tr['f1']:.3f}  F1 teste={m_te['f1']:.3f}")
    return curva
