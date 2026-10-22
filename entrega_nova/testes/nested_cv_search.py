"""Validação Cruzada Aninhada (Nested CV) para Seleção de Hiperparâmetros."""

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
    
    # Outer 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    estratos = [item["fora_da_taxonomia"] for item in fv]
    
    # Grid de hiperparâmetros para avaliação
    grid = [
        {"w_dense": 0.50, "w_sparse": 0.50, "T": 0.016},
        {"w_dense": 0.55, "w_sparse": 0.45, "T": 0.016},
        {"w_dense": 0.60, "w_sparse": 0.40, "T": 0.016},
        {"w_dense": 0.55, "w_sparse": 0.45, "T": 0.012},
        {"w_dense": 0.55, "w_sparse": 0.45, "T": 0.020},
    ]
    
    print("=== SELEÇÃO DE HIPERPARÂMETROS EM CADA FOLD (NESTED CV) ===")
    fold_choices = []
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(fv, estratos)):
        train_items = [fv[i] for i in train_idx]
        test_items = [fv[i] for i in test_idx]
        
        # Avaliar cada candidato na partição de treino (80 itens)
        best_brier = 999.0
        best_cfg = None
        
        for cfg in grid:
            motor.w_dense = cfg["w_dense"]
            motor.w_sparse = cfg["w_sparse"]
            motor.temperatura = cfg["T"]
            
            y_true_tr = []
            y_prob_tr = []
            for item in train_items:
                qid = item["id"]
                q_idx, q = q_map[qid]
                s = motor.calcular_scores_questao(q_idx, q)
                probs = motor.calibrar_probabilidades(s)
                n_est, p_est, _ = motor.maximizar_probabilidade_arvore(probs)
                acertou = avaliar_acerto({"decisao": "publicar", "tema_central_id": n_est}, item, arvore)
                y_true_tr.append(1 if acertou else 0)
                y_prob_tr.append(p_est)
                
            brier_tr = np.mean((np.array(y_prob_tr) - np.array(y_true_tr))**2)
            if brier_tr < best_brier:
                best_brier = brier_tr
                best_cfg = cfg
                
        fold_choices.append(best_cfg)
        print(f"Fold {fold+1}: Melhor Config Treino: w_dense={best_cfg['w_dense']:.2f}, w_sparse={best_cfg['w_sparse']:.2f}, T={best_cfg['T']:.3f} (Brier={best_brier:.4f})")
        
    print(f"\nConsenso dos Folds: {fold_choices[0]}")

if __name__ == "__main__":
    main()
