"""Experimento Leave-One-Out (LOO) 1-NN estritamente nos 100 rótulos ouro."""

import os
import sys
import numpy as np

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
    
    cache_path = os.path.join(BASE_DIR, "artefatos", "embeddings_cache.npz")
    data = np.load(cache_path, allow_pickle=True)
    q_dense = data["questoes_dense"]
    
    # Índices dos 100 itens da folha
    fv_idx = [q_map[item["id"]][0] for item in fv]
    fv_embeds = q_dense[fv_idx]
    fv_embeds = fv_embeds / np.maximum(np.linalg.norm(fv_embeds, axis=1, keepdims=True), 1e-12)
    
    # Matriz de similaridade cosseno 100x100
    sim_mat = fv_embeds @ fv_embeds.T
    np.fill_diagonal(sim_mat, -1.0) # Não comparar consigo mesmo
    
    print("=== LEAVE-ONE-OUT 1-NN STRICTLY ON 100 GOLD LABELS ===")
    
    for thresh in [0.0, 0.85, 0.90, 0.92, 0.94]:
        n_eval = 0
        n_correct = 0
        n_abstain = 0
        
        for i in range(100):
            item_test = fv[i]
            best_j = int(np.argmax(sim_mat[i]))
            best_sim = sim_mat[i, best_j]
            
            if best_sim >= thresh:
                n_eval += 1
                item_train = fv[best_j]
                if item_train["fora_da_taxonomia"] == 1:
                    dec = {"decisao": "fora", "tema_central_id": ""}
                else:
                    dec = {"decisao": "publicar", "tema_central_id": item_train["tema_central_id"]}
                if avaliar_acerto(dec, item_test, arvore):
                    n_correct += 1
            else:
                n_abstain += 1
                
        prec = (n_correct / n_eval * 100) if n_eval > 0 else 0
        print(f"Limiar sim >= {thresh:.2f} -> Avaliados: {n_eval}/100, Acertos: {n_correct}, Precisão: {prec:.2f}% (Abstenções: {n_abstain})")

if __name__ == "__main__":
    main()
