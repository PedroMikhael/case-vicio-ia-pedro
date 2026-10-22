"""Diagnóstico de P(fora) em todas as 659 questões."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao
from src.preparar import carregar_e_processar_questoes

def main():
    motor = MotorDecisao()
    questoes = carregar_e_processar_questoes()
    
    p_fora_list = []
    motivos = []
    
    for idx, q in enumerate(questoes):
        scores = motor.calcular_scores_questao(idx, q)
        probs = motor.calibrar_probabilidades(scores)
        n_est, p_est, _ = motor.maximizar_probabilidade_arvore(probs)
        p_fora, motivo = motor.estimar_probabilidade_fora(idx, q, scores, p_est)
        p_fora_list.append((q["id"], p_fora, motivo, p_est))
        if p_fora >= 0.40:
            motivos.append((q["id"], p_fora, motivo, p_est))
            
    print(f"Total com p_fora >= 0.40: {len(motivos)}")
    for qid, pf, mot, pest in motivos[:15]:
        print(f"  {qid}: P(fora)={pf:.2f}, P*(acerto)={pest:.2f}, Motivo={mot}")
        
    p_vals = [x[1] for x in p_fora_list]
    import numpy as np
    print(f"Max P(fora): {np.max(p_vals):.4f}, Mean: {np.mean(p_vals):.4f}")

if __name__ == "__main__":
    main()
