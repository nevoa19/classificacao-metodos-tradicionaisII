# Classificação com métodos tradicionais II

Mini-projeto 02 (IA): árvore de decisão, floresta aleatória e XGBoost nas mesmas duas bases do mini-projeto 01.

- CDC Diabetes Health Indicators (OpenML 46841), alvo `Diabetes_binary`
- Predict Students' Dropout and Academic Success (Kaggle, `thedevastator/higher-education-predictors-of-student-retention`), alvo `Target` convertido em evasão (Dropout = 1) x não evasão (Graduate/Enrolled = 0)

## Requisitos
- Python 3.10 ou superior
- numpy, pandas, scikit-learn, scipy, matplotlib, xgboost

    pip install -r requirements.txt

## Como executar
    python classificacao.py

Para rodar só uma base, alterar `RODAR_CDC` ou `RODAR_ESTUDANTES` para `False` no início do arquivo.
A base de estudantes leva poucos minutos; o CDC demora.

## Arquivos
- `classificacao.py`: roda o experimento das duas bases e gera tabelas e figuras
- `dados.py`: leitura das bases, divisão treino/teste e subamostragem 
- `modelos.py`: modelos e grades de hiperparâmetros
- `comum.py`: busca de hiperparâmetros, reajuste no treino inteiro, métricas e teste de estabilidade
- `arvore.py`: árvore de decisão, curva de profundidade, árvore em texto e estabilidade
- `floresta.py`: floresta aleatória, curva métrica x nº de árvores e estabilidade
- `boosting.py`: XGBoost 
- `metricas.py`: métricas, IC por bootstrap e McNemar 
- `graficos.py`: figuras
- `tabelas.py`: tabelas em LaTeX

## Dados
Os dois arquivos ficam em `data/`.

## Protocolo
- treino/teste estratificada 80/20, semente 42
- validação cruzada estratificada só no treino (3 folds no CDC, 5 nos estudantes), critério F1 da classe positiva
- árvore: GridSearchCV em `criterion` e `max_depth`
- floresta: RandomizedSearchCV (20 combinações) em `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`
- XGBoost: RandomizedSearchCV (30 combinações) em `n_estimators`, `learning_rate`, `max_depth`, `subsample`
- desbalanceamento: `class_weight="balanced"` na árvore e na floresta; `scale_pos_weight` = negativos/positivos no XGBoost
- CDC: busca da floresta e do XGBoost em subamostra estratificada de 30.000 exemplos do treino; árvore com o treino inteiro
- depois da busca, cada modelo é reajustado com os melhores hiperparâmetros no treino inteiro 
- curva da floresta: demais hiperparâmetros fixos no melhor valor, variando só `n_estimators` (com `warm_start`)
- estabilidade: 5 rodadas removendo 5% do treino ao acaso; mede o F1 no teste e a fração do teste em que as previsões mudam
- IC 95% do F1 por bootstrap (1.000 reamostragens) e teste de McNemar exato 
- comparação com o projeto 01 

## Saídas
Em `resultados/`:
- `tabelas/`: `*_hiperparametros`, `*_metricas` (treino e teste), `*_significancia_tempo`, `*_estabilidade`, `juntas_comparacao`
- `figuras/`: matrizes de confusão, árvore (3 níveis), F1 x profundidade, F1 x nº de árvores, importâncias dos atributos e painel comparativo
- `*_arvore.txt`: melhor árvore em texto (até profundidade 4)
- `resultados.json`: todos os números
