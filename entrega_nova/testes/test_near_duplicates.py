"""Verificar quase-duplicatas entre as 559 não-rotuladas e as 100 da folha-verdade."""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade

def main():
    questoes = carregar_e_processar_questoes()
    fv = carregar_folha_verdade()
    fv_ids = {item["id"] for item in fv}
    fv_map = {item["id"]: item for item in fv}
    
    cache_path = os.path.join(BASE_DIR, "artefatos", "embeddings_cache.npz")
    data = np.load(cache_path, allow_pickle=True)
    q_dense = data["questoes_dense"]
    q_norm = q_dense / np.maximum(np.linalg.norm(q_dense, axis=1, keepdims=True), 1e-12)
    
    q_fv_idx = [i for i, q in enumerate(questoes) if q["id"] in fv_ids]
    q_unlab_idx = [i for i, q in enumerate(questoes) if q["id"] not in fv_ids]
    
    fv_embeds = q_norm[q_fv_idx]
    unlab_embeds = q_norm[q_unlab_idx]
    
    sims = unlab_embeds @ fv_embeds.T # (559, 100)
    
    print(f"Max sim unlabeled -> gold: {np.max(sims):.4f}")
    
    # Quantas questões têm similaridade >= 0.95 com alguma questão ouro?
    altas_sims = []
    for u_idx_rel, u_idx in enumerate(q_unlab_idx):
        max_sim = np.max(sims[u_idx_rel])
        best_gold_rel = np.argmax(sims[u_idx_rel])
        gold_id = questoes[q_fv_idx[best_gold_rel]]["id"]
        if max_sim >= 0.95:
            altas_sims.append((questoes[u_idx]["id"], gold_id, max_sim, fv_map[gold_id]["tema_central_id"]))
            
    print(f"Total de quase-duplicatas (sim >= 0.95): {len(altas_sims)}")
    for u_id, g_id, s, tid in altas_sims[:10]:
        print(f"  {u_id} <-> {g_id}: sim={s:.4f}, tema={tid}")

if __name__ == "__main__":
    main()
