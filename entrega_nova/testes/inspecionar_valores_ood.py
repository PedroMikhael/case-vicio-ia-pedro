"""Inspecionar valores reais dos scores para OOD."""

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
    fv_map = {item["id"]: item for item in fv}
    
    q_map = {q["id"]: (i, q) for i, q in enumerate(questoes)}
    
    print("=== SCORES DOS 8 ITENS FORA DA TAXONOMIA NA FOLHA-VERDADE ===")
    for item in fv:
        if item["fora_da_taxonomia"] == 1:
            qid = item["id"]
            idx, q = q_map[qid]
            dense_max = np.max(motor.dense_sim[idx])
            dense_min = np.min(motor.dense_sim[idx])
            inat_max = np.max(motor.inativos_dense_sim[idx]) if motor.inativos_dense_sim.shape[1] > 0 else 0
            
            # Disciplina / assunto
            disc = q.get("disciplina_origem")
            assunto = q.get("assunto_origem")
            
            # Scores da árvore
            s = motor.calcular_scores_questao(idx, q)
            probs = motor.calibrar_probabilidades(s)
            n_est, p_est, _ = motor.maximizar_probabilidade_arvore(probs)
            
            print(f"{qid}: dense_max={dense_max:.4f}, inat_max={inat_max:.4f}, diff={inat_max - dense_max:.4f}, p_est={p_est:.4f} | disc='{disc}', ass='{assunto}'")
            
    print("\n=== SCORES DAS 10 QUESTÕES MILITARES NO LOTE RECENTE ===")
    militares = [q for q in questoes if "militar" in (q.get("disciplina_origem") or "").lower() or "militar" in (q.get("assunto_origem") or "").lower()]
    print(f"Total identificados como militares: {len(militares)}")
    for q in militares[:10]:
        qid = q["id"]
        idx = q_map[qid][0]
        dense_max = np.max(motor.dense_sim[idx])
        s = motor.calcular_scores_questao(idx, q)
        probs = motor.calibrar_probabilidades(s)
        n_est, p_est, _ = motor.maximizar_probabilidade_arvore(probs)
        print(f"{qid}: dense_max={dense_max:.4f}, p_est={p_est:.4f} | disc='{q.get('disciplina_origem')}', ass='{q.get('assunto_origem')}'")

if __name__ == "__main__":
    main()
