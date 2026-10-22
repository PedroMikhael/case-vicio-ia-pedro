"""Teste de Enriquecimento de Rótulos Ouro: Pseudo-rotulação e Concordância com Exemplares."""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao
from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade
from src.validar_cv import avaliar_acerto

def main():
    motor = MotorDecisao()
    arvore = motor.arvore
    questoes = carregar_e_processar_questoes()
    q_map = {q["id"]: (i, q) for i, q in enumerate(questoes)}
    fv = carregar_folha_verdade()
    fv_map = {item["id"]: item for item in fv}
    
    # Vamos avaliar para as 100 questões da folha:
    # O que o motor_decisao prediz (n*, P(n*))?
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    estratos = [item["fora_da_taxonomia"] for item in fv]
    
    # Carregar embeddings
    cache_path = os.path.join(BASE_DIR, "artefatos", "embeddings_cache.npz")
    data = np.load(cache_path, allow_pickle=True)
    q_dense = data["questoes_dense"]
    
    # Normalizar
    norms = np.linalg.norm(q_dense, axis=1, keepdims=True)
    q_dense = q_dense / np.maximum(norms, 1e-12)
    
    # Analisar todas as 100 questões da folha:
    resultados = []
    for idx, item in enumerate(fv):
        qid = item["id"]
        q_idx, questao = q_map[qid]
        s_hib = motor.calcular_scores_questao(q_idx, questao)
        probs = motor.calibrar_probabilidades(s_hib)
        n_est, p_est, p_acum = motor.maximizar_probabilidade_arvore(probs)
        dec = {"decisao": "publicar", "tema_central_id": n_est}
        acertou = avaliar_acerto(dec, item, arvore)
        resultados.append({
            "id": qid,
            "q_idx": q_idx,
            "n_est": n_est,
            "p_est": p_est,
            "acertou": acertou,
            "fora": item["fora_da_taxonomia"]
        })
        
    print(f"Total avaliados: {len(resultados)}")
    # Ordenar por p_est descrescente:
    resultados_ord = sorted(resultados, key=lambda x: x["p_est"], reverse=True)
    
    print("\n--- CURVA DE CONFIANÇA CUMULATIVA ---")
    cum_pub = 0
    cum_corr = 0
    for r in resultados_ord:
        cum_pub += 1
        if r["acertou"]: cum_corr += 1
        prec = cum_corr / cum_pub * 100
        if cum_pub in [5, 10, 15, 20, 23, 25, 30, 40, 50, 60, 70, 80, 90, 100]:
            print(f"Top {cum_pub:3d} (P >= {r['p_est']:.4f}) -> Acertos: {cum_corr:3d} | Precisão: {prec:5.2f}% | Fora: {r['fora']}")

if __name__ == "__main__":
    main()
