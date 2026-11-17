from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from dados import SEMENTE

MODELOS = ["arvore", "floresta", "xgboost"]

NOMES = {}
NOMES["trivial"] = "Trivial (classe majoritária)"
NOMES["arvore"] = "Árvore de decisão"
NOMES["floresta"] = "Floresta aleatória"
NOMES["xgboost"] = "XGBoost"

NOMES_MINI1 = {}
NOMES_MINI1["svm_linear"] = "SVM linear"
NOMES_MINI1["svm_rbf"] = "SVM RBF"
NOMES_MINI1["logreg"] = "Reg. logística"
NOMES_MINI1["knn"] = "k-NN"

GRADES = {}
GRADES["arvore"] = {"criterion": ["gini", "entropy"],
                    "max_depth": [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, None]}
GRADES["floresta"] = {"n_estimators": [100, 200, 300, 500],
                      "max_depth": [5, 10, 15, 20, None],
                      "max_features": ["sqrt", "log2", 0.3, 0.5],
                      "min_samples_leaf": [1, 2, 5, 10, 20]}
GRADES["xgboost"] = {"n_estimators": [100, 200, 400, 800],
                     "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.3],
                     "max_depth": [2, 3, 4, 6, 8],
                     "subsample": [0.6, 0.8, 1.0]}

ALEATORIA = {}
ALEATORIA["arvore"] = False
ALEATORIA["floresta"] = True
ALEATORIA["xgboost"] = True

N_ITER = {}
N_ITER["floresta"] = 20
N_ITER["xgboost"] = 30

def peso_positivos(y):
    neg = 0
    pos = 0
    for v in y:
        if v == 1:
            pos += 1
        else:
            neg += 1
    return neg / pos

def criar_modelo(nome, y=None):
    if nome == "arvore":
        return DecisionTreeClassifier(class_weight="balanced", random_state=SEMENTE)
    if nome == "floresta":
        return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=SEMENTE)
    if nome == "xgboost":
        peso = 1.0
        if y is not None:
            peso = peso_positivos(y)
        return XGBClassifier(eval_metric="logloss", tree_method="hist", scale_pos_weight=peso,
                             n_jobs=-1, random_state=SEMENTE)

def criar_com_parametros(nome, parametros, y=None):
    modelo = criar_modelo(nome, y)
    for k in parametros:
        setattr(modelo, k, parametros[k])
    return modelo
