"""Grid Search e Validação Aninhada de Hiperparâmetros."""

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
    
    # Grid de pesquisa
    # w_dense: 0.60, 0.70, 0.80
    # w_sparse: 1 - w_dense
    # T: 0.003, 0.005, 0.008, 0.010
    print("--- BUSCA EM GRADE (GRID SEARCH) DE HIPERPARÂMETROS ---")
    
    for w_d in [0.60, 0.70, 0.80]:
        w_s = 1.0 - w_d
        motor.w_dense = w_d
        motor.w_sparse = w_s
        
        for T in [0.003, 0.005, 0.008, 0.010]:
            # Avaliar ECE e precisão @ tau=0.92
            y_true = []
            y_prob = []
            
            for item in fv:
                qid = item["id"]
                q_idx, questao = q_map[qid]
                s_hib = motor.calcular_scores_questao(q_idx, questao)
                probs = motor.calibrar_probabilidades(s_hib, temperatura=T)
                n_est, p_est, _ = motor.maximizar_probabilidade_arvore(probs)
                dec = {"decisao": "publicar", "tema_central_id": n_est}
                acertou = avaliar_acerto(dec, item, arvore)
                y_true.append(1 if acertou else 0)
                y_prob.append(float(p_est))
                
            y_true = np.array(y_true)
            y_prob = np.array(y_prob)
            
            # Precisão e cobertura com tau=0.92
            mask_pub = y_prob >= 0.92
            n_pub = np.sum(mask_pub)
            prec = (np.sum(y_true[mask_pub]) / n_pub * 100) if n_pub > 0 else 0
            
            # Brier
            brier = np.mean((y_prob - y_true) ** 2)
            
            print(f"w_dense={w_d:.2f}, w_sparse={w_s:.2f}, T={T:.4f} -> Pub={n_pub:2d}, Acertos={np.sum(y_true[mask_pub]):2d}, Prec={prec:5.2f}%, Brier={brier:.4f}")

if __name__ == "__main__":
    main()
