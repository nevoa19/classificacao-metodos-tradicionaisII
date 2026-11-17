import os
import json
from dados import ler_cdc, ler_estudantes, dividir
from modelos import MODELOS, NOMES, NOMES_MINI1
from metricas import N_BOOTSTRAP, calcular_metricas, intervalo_f1, mcnemar
from tabelas import numero_br, salvar_tabela
from arvore import rodar_arvore
from floresta import rodar_floresta
from boosting import rodar_xgboost
from graficos import grafico_matrizes, grafico_arvore, grafico_profundidade, grafico_n_arvores, grafico_importancias, grafico_painel

RODAR_CDC = True
RODAR_ESTUDANTES = True

RODADAS_ESTAB = 5

ARQ_MINI1 = "../classificacao-metodos-tradicionais/resultados/resultados.json"

def texto_parametros(params):
    texto = ""
    for k in params:
        if texto != "":
            texto = texto + ", "
        v = params[k]
        if v is None:
            v = "None"
        texto = texto + k.replace("_", "\\_") + "=" + numero_br(v, 2).replace("_", "\\_")
    return texto

def rodar_base(tag, titulo, X, y, classes, folds, sub_busca):
    print(f"\n=== {titulo}")
    Xtr, Xte, ytr, yte = dividir(X, y)
    print(f"treino={len(Xtr)}  teste={len(Xte)}")

    resultados = {}
    previsoes = {}
    treinados = {}

    classe_maj = 0
    if ytr.sum() > len(ytr) / 2:
        classe_maj = 1
    chute_treino = []
    i = 0
    while i < len(ytr):
        chute_treino.append(classe_maj)
        i += 1
    chute_teste = []
    i = 0
    while i < len(yte):
        chute_teste.append(classe_maj)
        i += 1
    resultados["trivial"] = {}
    resultados["trivial"]["treino"] = calcular_metricas(ytr, chute_treino)
    resultados["trivial"]["teste"] = calcular_metricas(yte, chute_teste)

    print("arvore...")
    r, modelo, prev = rodar_arvore(tag, Xtr, ytr, Xte, yte, folds, None, RODADAS_ESTAB)
    resultados["arvore"] = r
    treinados["arvore"] = modelo
    previsoes["arvore"] = prev

    print("floresta...")
    r, modelo, prev = rodar_floresta(Xtr, ytr, Xte, yte, folds, sub_busca, RODADAS_ESTAB)
    resultados["floresta"] = r
    treinados["floresta"] = modelo
    previsoes["floresta"] = prev

    print("xgboost...")
    r, modelo, prev = rodar_xgboost(Xtr, ytr, Xte, yte, folds, sub_busca)
    resultados["xgboost"] = r
    treinados["xgboost"] = modelo
    previsoes["xgboost"] = prev

    for m in MODELOS:
        r = resultados[m]
        print(f"{NOMES[m]:20s} {r['parametros']}  F1 vc={r['f1_vc']:.3f}  F1 treino={r['treino']['f1']:.3f}  F1 teste={r['teste']['f1']:.3f}  acc teste={r['teste']['acuracia']:.3f}  (busca {r['segundos_busca']}s, ajuste {r['segundos_ajuste']}s)")

    melhor = MODELOS[0]
    for m in MODELOS:
        if resultados[m]["teste"]["f1"] > resultados[melhor]["teste"]["f1"]:
            melhor = m
    for m in MODELOS:
        resultados[m]["ic95_f1"] = intervalo_f1(yte, previsoes[m])
        if m == melhor:
            resultados[m]["p_mcnemar"] = None
        else:
            resultados[m]["p_mcnemar"] = mcnemar(yte, previsoes[melhor], previsoes[m])
    p_xgb_rf = mcnemar(yte, previsoes["xgboost"], previsoes["floresta"])

    linhas = []
    for m in MODELOS:
        r = resultados[m]
        linhas.append([NOMES[m], texto_parametros(r["parametros"]), numero_br(r["f1_vc"])])
    salvar_tabela(tag + "_hiperparametros", ["Modelo", "Hiperparâmetros", "F1 (VC)"], linhas,
                  titulo + ": hiperparâmetros escolhidos e F1 médio na validação cruzada.", "tab:" + tag + "_hiper", "llc")

    linhas = []
    for m in ["trivial"] + MODELOS:
        for conj in ["treino", "teste"]:
            t = resultados[m][conj]
            nome = NOMES[m]
            if conj == "teste":
                nome = ""
            linhas.append([nome, conj, numero_br(t["acuracia"]), numero_br(t["precisao"]),
                           numero_br(t["revocacao"]), numero_br(t["f1"])])
    salvar_tabela(tag + "_metricas", ["Modelo", "Conjunto", "Acur.", "Prec.", "Rev.", "F1"], linhas,
                  titulo + ": métricas no treino e no teste (classe positiva: " + classes[1].lower() + ").", "tab:" + tag + "_metricas", "llcccc")

    linhas = []
    for m in MODELOS:
        r = resultados[m]
        ic = r["ic95_f1"]
        p = r["p_mcnemar"]
        if p is None:
            texto_p = "--"
        elif p < 0.0001:
            texto_p = "$<10^{-4}$"
        else:
            texto_p = numero_br(p, 4)
        linhas.append([NOMES[m], "[" + numero_br(ic[0]) + "; " + numero_br(ic[1]) + "]", texto_p,
                       numero_br(r["segundos_busca"], 1), numero_br(r["segundos_ajuste"], 2)])
    salvar_tabela(tag + "_significancia_tempo", ["Modelo", "IC 95\\% do F1", "$p$ McNemar", "Busca (s)", "Ajuste (s)"], linhas,
                  titulo + ": IC 95\\% do F1 (bootstrap, " + str(N_BOOTSTRAP) + " reamostragens), teste de McNemar contra o melhor modelo (" + NOMES[melhor] +
                  ") e tempos de treino (busca de hiperparâmetros e um ajuste com os melhores valores no treino inteiro).",
                  "tab:" + tag + "_sig")

    linhas = []
    e = resultados["arvore"]["estabilidade_sem_limite"]
    linhas.append(["Árvore (sem limite)", numero_br(e["f1_media"]) + " $\\pm$ " + numero_br(e["f1_dp"]), numero_br(e["discordancia_media"] * 100, 1) + "\\%"])
    e = resultados["arvore"]["estabilidade"]
    linhas.append(["Árvore (melhor)", numero_br(e["f1_media"]) + " $\\pm$ " + numero_br(e["f1_dp"]), numero_br(e["discordancia_media"] * 100, 1) + "\\%"])
    e = resultados["floresta"]["estabilidade"]
    linhas.append(["Floresta (melhor)", numero_br(e["f1_media"]) + " $\\pm$ " + numero_br(e["f1_dp"]), numero_br(e["discordancia_media"] * 100, 1) + "\\%"])
    salvar_tabela(tag + "_estabilidade", ["Modelo", "F1 teste", "Previsões diferentes"], linhas,
                  titulo + ": efeito de remover 5\\% do treino ao acaso (" + str(RODADAS_ESTAB) +
                  " rodadas). Previsões diferentes: fração média do teste em que dois modelos de rodadas distintas discordam.",
                  "tab:" + tag + "_estab")

    grafico_matrizes(resultados, tag, classes)
    grafico_arvore(treinados["arvore"], X.columns, tag, classes)
    grafico_profundidade(resultados["arvore"]["curva_profundidade"], resultados["arvore"]["parametros"]["max_depth"], tag)
    grafico_n_arvores(resultados["floresta"]["curva"], tag)
    grafico_importancias(treinados, list(X.columns), tag, 10)

    saida = {}
    saida["titulo"] = titulo
    saida["n"] = len(X)
    saida["n_treino"] = len(Xtr)
    saida["n_teste"] = len(Xte)
    saida["melhor_modelo"] = melhor
    saida["p_mcnemar_xgboost_floresta"] = p_xgb_rf
    saida["modelos"] = resultados
    return saida

def tabela_comparacao(todos, mini1):
    tags = list(todos.keys())
    cab = ["Modelo", "Mini-projeto"]
    for t in tags:
        cab.append(todos[t]["titulo"] + " (F1)")
    linhas = []
    if mini1 is not None:
        for m in NOMES_MINI1:
            linha = [NOMES_MINI1[m], "I"]
            for t in tags:
                if t in mini1 and m in mini1[t]["modelos"]:
                    linha.append(numero_br(mini1[t]["modelos"][m]["teste"]["f1"]))
                else:
                    linha.append("--")
            linhas.append(linha)
    for m in MODELOS:
        linha = [NOMES[m], "II"]
        for t in tags:
            linha.append(numero_br(todos[t]["modelos"][m]["teste"]["f1"]))
        linhas.append(linha)
    salvar_tabela("juntas_comparacao", cab, linhas,
                  "F1 no conjunto de teste de todos os modelos dos mini-projetos I e II (mesma divisão treino/teste).",
                  "tab:comparacao")

if __name__ == "__main__":
    if not os.path.exists("resultados"):
        os.makedirs("resultados")

    todos = {}
    if os.path.exists("resultados/resultados.json"):
        arq = open("resultados/resultados.json", encoding="utf-8")
        todos = json.load(arq)
        arq.close()

    if RODAR_CDC:
        X, y = ler_cdc()
        todos["cdc"] = rodar_base("cdc", "CDC Diabetes", X, y, ["Sem diabetes", "Com diabetes"], 3, 30000)

    if RODAR_ESTUDANTES:
        X, y = ler_estudantes()
        todos["estudantes"] = rodar_base("estudantes", "Evasão de estudantes", X, y, ["Não evasão", "Evasão"], 5, None)

    ordem = {}
    for t in ["cdc", "estudantes"]:
        if t in todos:
            ordem[t] = todos[t]

    mini1 = None
    if os.path.exists(ARQ_MINI1):
        arq = open(ARQ_MINI1, encoding="utf-8")
        mini1 = json.load(arq)
        arq.close()
    else:
        print("aviso: resultados do mini-projeto I não encontrados em " + ARQ_MINI1)

    tabela_comparacao(ordem, mini1)
    grafico_painel(ordem, mini1)

    arq = open("resultados/resultados.json", "w", encoding="utf-8")
    json.dump(ordem, arq, indent=2, ensure_ascii=False, default=str)
    arq.close()
    print("\nfiguras e tabelas salvas na pasta resultados/")
