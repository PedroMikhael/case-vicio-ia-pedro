"""Experimento de kNN / Exemplar Bank sobre os rótulos ouro na CV 5x5."""

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
    
    # Obter embeddings das 100 questões da folha
    # motor.dense_sim é (659, 371), mas temos os embeddings em cache?
    cache_path = os.path.join(BASE_DIR, "artefatos", "embeddings_cache.npz")
    data = np.load(cache_path, allow_pickle=True)
    q_dense = data["questoes_dense"] # (659, 768)
    
    fv_indices = [q_map[item["id"]][0] for item in fv]
    fv_labels = [item["tema_central_id"] if item["fora_da_taxonomia"] == 0 else "FORA" for item in fv]
    
    print(f"Total fv indices: {len(fv_indices)}, shape q_dense: {q_dense.shape}")
    
    # Calcular similaridade entre questões ouro:
    fv_embeds = q_dense[fv_indices] # (100, 768)
    # normalizados?
    norms = np.linalg.norm(fv_embeds, axis=1, keepdims=True)
    fv_embeds = fv_embeds / np.maximum(norms, 1e-12)
    
    sim_mat = np.dot(fv_embeds, fv_embeds.T)
    np.fill_diagonal(sim_mat, 0) # ignorar self
    
    print(f"Max sim entre pares ouro: {np.max(sim_mat):.3f}, Mean max sim: {np.mean(np.max(sim_mat, axis=1)):.3f}")
    
    # Testar kNN out-of-fold com 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    estratos = [item["fora_da_taxonomia"] for item in fv]
    
    for k_neighbors in [1, 3]:
        for thresh_knn in [0.85, 0.88, 0.90, 0.92]:
            pub_count = 0
            correct_count = 0
            
            for train_idx, test_idx in skf.split(fv, estratos):
                train_embeds = fv_embeds[train_idx]
                train_items = [fv[i] for i in train_idx]
                
                for ti in test_idx:
                    test_emb = fv_embeds[ti]
                    test_item = fv[ti]
                    
                    # Similaridade com o banco de treino
                    sims = np.dot(train_embeds, test_emb)
                    best_idx = np.argmax(sims)
                    best_sim = sims[best_idx]
                    
                    if best_sim >= thresh_knn:
                        exemplar = train_items[best_idx]
                        if exemplar["fora_da_taxonomia"] == 0:
                            pred_tid = exemplar["tema_central_id"]
                            dec = {"decisao": "publicar", "tema_central_id": pred_tid}
                            pub_count += 1
                            if avaliar_acerto(dec, test_item, arvore):
                                correct_count += 1
            
            prec = (correct_count / pub_count * 100) if pub_count > 0 else 0
            print(f"kNN k={k_neighbors}, thresh={thresh_knn:.2f} -> Pub={pub_count}/100, Correct={correct_count}, Prec={prec:.1f}%")

if __name__ == "__main__":
    main()
