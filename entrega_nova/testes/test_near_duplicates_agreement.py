"""Verificar a decisão atual do motor sobre as 65 quase-duplicatas."""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao
from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade

def main():
    motor = MotorDecisao()
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
    
    sims = unlab_embeds @ fv_embeds.T
    
    pub_ja = 0
    rev_ja = 0
    concordam = 0
    
    for u_idx_rel, u_idx in enumerate(q_unlab_idx):
        max_sim = np.max(sims[u_idx_rel])
        best_gold_rel = np.argmax(sims[u_idx_rel])
        gold_id = questoes[q_fv_idx[best_gold_rel]]["id"]
        
        if max_sim >= 0.95:
            gold_item = fv_map[gold_id]
            q = questoes[u_idx]
            dec = motor.decidir_questao(u_idx, q)
            
            if dec["decisao"] == "publicar":
                pub_ja += 1
            else:
                rev_ja += 1
                
            # Verifica se o tema sugerido em revisar ou publicar é do mesmo ramo do ouro:
            gold_tid = gold_item["tema_central_id"]
            if gold_item["fora_da_taxonomia"] == 0 and dec["tema_central_id"] == gold_tid:
                concordam += 1
                
    print(f"Das 65 quase-duplicatas: {pub_ja} já estão em 'publicar', {rev_ja} estão em 'revisar'.")
    print(f"Concordância exata com o tema do ouro: {concordam}/{len(q_unlab_idx)}")

if __name__ == "__main__":
    main()
