"""Cálculo formal de Calibração: ECE, Brier Score e Reliability Diagram."""

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
    
    y_true = []
    y_prob = []
    
    for item in fv:
        qid = item["id"]
        q_idx, questao = q_map[qid]
        s_hib = motor.calcular_scores_questao(q_idx, questao)
        probs = motor.calibrar_probabilidades(s_hib)
        n_est, p_est, p_acum = motor.maximizar_probabilidade_arvore(probs)
        dec = {"decisao": "publicar", "tema_central_id": n_est}
        acertou = avaliar_acerto(dec, item, arvore)
        y_true.append(1 if acertou else 0)
        y_prob.append(float(p_est))
        
    y_true = np.array(y_true)
    y_prob = np.array(y_prob)
    
    # 1. Brier Score
    brier = np.mean((y_prob - y_true) ** 2)
    print(f"Brier Score Global (N=100): {brier:.4f}")
    
    # 2. Bins para ECE (Expected Calibration Error) e Reliability Diagram
    # Definindo 5 bins padrão ou por faixas
    bin_edges = [0.0, 0.3, 0.5, 0.7, 0.85, 0.92, 1.00]
    ece = 0.0
    total_n = len(y_true)
    
    print("\n--- TABELA DO DIAGRAMA DE CONFIABILIDADE (RELIABILITY DIAGRAM) ---")
    print(f"{'Faixa de Confiança':<25} | {'N Itens':<8} | {'Conf Média':<12} | {'Acurácia Real':<14} | {'Gap (|Conf - Acc|)'}")
    print("-" * 75)
    
    tabela_calibracao = []
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (y_prob >= low) & (y_prob < high) if high < 1.0 else (y_prob >= low) & (y_prob <= high)
        n_bin = np.sum(mask)
        if n_bin > 0:
            conf_media = np.mean(y_prob[mask])
            acc_real = np.mean(y_true[mask])
            gap = abs(conf_media - acc_real)
            ece += (n_bin / total_n) * gap
            print(f"[{low:.2f}, {high:.2f}){' ' if high < 1.0 else ']'}{'':<14} | {n_bin:<8} | {conf_media*100:9.2f}%   | {acc_real*100:10.2f}%    | {gap*100:6.2f}%")
            tabela_calibracao.append({
                "faixa": f"[{low:.2f}, {high:.2f}]",
                "n": int(n_bin),
                "conf_media": float(conf_media),
                "acc_real": float(acc_real),
                "gap": float(gap)
            })
            
    print("-" * 75)
    print(f"Expected Calibration Error (ECE): {ece*100:.2f}% ({ece:.4f})")

if __name__ == "__main__":
    main()
