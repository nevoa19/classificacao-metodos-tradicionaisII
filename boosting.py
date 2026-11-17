from modelos import peso_positivos
from comum import buscar_e_avaliar

def rodar_xgboost(Xtr, ytr, Xte, yte, folds, sub_busca):
    r, modelo, prev = buscar_e_avaliar("xgboost", Xtr, ytr, Xte, yte, folds, sub_busca)
    r["scale_pos_weight"] = peso_positivos(ytr)
    return r, modelo, prev
