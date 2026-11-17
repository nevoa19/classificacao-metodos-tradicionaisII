import os
from sklearn.tree import export_text
from modelos import criar_com_parametros
from metricas import calcular_metricas
from comum import buscar_e_avaliar, estabilidade

LISTA_PROFUNDIDADES = [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]

def rodar_arvore(tag, Xtr, ytr, Xte, yte, folds, sub_busca, rodadas_estab):
    r, modelo, prev = buscar_e_avaliar("arvore", Xtr, ytr, Xte, yte, folds, sub_busca)
    r["profundidade_real"] = int(modelo.get_depth())
    r["folhas"] = int(modelo.get_n_leaves())

    if not os.path.exists("resultados"):
        os.makedirs("resultados")
    texto = export_text(modelo, feature_names=list(Xtr.columns), max_depth=4)
    arq = open("resultados/" + tag + "_arvore.txt", "w", encoding="utf-8")
    arq.write(texto)
    arq.close()

    curva = {}
    curva["profundidade"] = []
    curva["treino"] = []
    curva["teste"] = []
    curva["folhas"] = []
    for p in LISTA_PROFUNDIDADES:
        params = {}
        params["criterion"] = r["parametros"]["criterion"]
        params["max_depth"] = p
        m = criar_com_parametros("arvore", params)
        m.fit(Xtr, ytr)
        curva["profundidade"].append(p)
        curva["treino"].append(calcular_metricas(ytr, m.predict(Xtr))["f1"])
        curva["teste"].append(calcular_metricas(yte, m.predict(Xte))["f1"])
        curva["folhas"].append(int(m.get_n_leaves()))
    r["curva_profundidade"] = curva

    params = {}
    params["criterion"] = r["parametros"]["criterion"]
    params["max_depth"] = None
    m = criar_com_parametros("arvore", params)
    m.fit(Xtr, ytr)
    r["sem_limite"] = {}
    r["sem_limite"]["profundidade"] = int(m.get_depth())
    r["sem_limite"]["folhas"] = int(m.get_n_leaves())
    r["sem_limite"]["treino"] = calcular_metricas(ytr, m.predict(Xtr))
    r["sem_limite"]["teste"] = calcular_metricas(yte, m.predict(Xte))

    r["estabilidade_sem_limite"] = estabilidade("arvore", params, Xtr, ytr, Xte, yte, rodadas_estab, 0.05)
    r["estabilidade"] = estabilidade("arvore", r["parametros"], Xtr, ytr, Xte, yte, rodadas_estab, 0.05)

    return r, modelo, prev
