"""Teste de Centróides de Exemplares da Folha-Verdade para Enriquecimento dos Temas."""

import os
import sys
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import defaultdict

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
    t_dense = data["temas_dense"]
    
    # Normalizar
    q_norm = q_dense / np.maximum(np.linalg.norm(q_dense, axis=1, keepdims=True), 1e-12)
    t_norm = t_dense / np.maximum(np.linalg.norm(t_dense, axis=1, keepdims=True), 1e-12)
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    estratos = [item["fora_da_taxonomia"] for item in fv]
    
    # Testar blending dos centróides de treino nos temas
    for alpha in [0.0, 0.1, 0.2, 0.3]:
        # alpha é o peso dos centróides de questões ouro no embedding do tema
        acertos_oof = 0
        pub_oof = 0
        
        for train_idx, test_idx in skf.split(fv, estratos):
            train_items = [fv[i] for i in train_idx]
            test_items = [fv[i] for i in test_idx]
            
            # Agrupar embeddings de treino por tema_id
            tema_exemplares = defaultdict(list)
            for item in train_items:
                if item["fora_da_taxonomia"] == 0:
                    tid = item["tema_central_id"]
                    q_idx = q_map[item["id"]][0]
                    tema_exemplares[tid].append(q_norm[q_idx])
                    
            # Criar t_dense_augmented
            t_aug = t_norm.copy()
            if alpha > 0.0:
                for tid, list_emb in tema_exemplares.items():
                    if tid in motor.tema_idx_map:
                        t_idx = motor.tema_idx_map[tid]
                        mean_emb = np.mean(list_emb, axis=0)
                        mean_emb = mean_emb / np.linalg.norm(mean_emb)
                        blended = (1.0 - alpha) * t_aug[t_idx] + alpha * mean_emb
                        t_aug[t_idx] = blended / np.linalg.norm(blended)
                        
            # Avaliar teste
            for item in test_items:
                qid = item["id"]
                q_idx = q_map[qid][0]
                q_emb = q_norm[q_idx]
                
                # Dense sim com temas aumentados
                dense_sim_aug = np.dot(t_aug, q_emb)
                sparse_sim = motor.sparse_sim[q_idx]
                
                scores = 0.55 * dense_sim_aug + 0.45 * sparse_sim
                probs = motor.calibrar_probabilidades(scores, temperatura=0.016)
                n_est, p_est, _ = motor.maximizar_probabilidade_arvore(probs)
                
                if p_est >= 0.92:
                    pub_oof += 1
                    dec = {"decisao": "publicar", "tema_central_id": n_est}
                    if avaliar_acerto(dec, item, arvore):
                        acertos_oof += 1
                        
        prec = (acertos_oof / pub_oof * 100) if pub_oof > 0 else 0
        print(f"Alpha={alpha:.1f} -> Pub={pub_oof}/100, Acertos={acertos_oof}, Precisão={prec:.2f}%")

if __name__ == "__main__":
    main()
