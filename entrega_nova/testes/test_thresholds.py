"""Verificar métricas exatas para limiares entre 0.90 e 0.98."""

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
    
    # Avaliar itens
    itens_eval = []
    for item in fv:
        qid = item["id"]
        q_idx, questao = q_map[qid]
        s_hib = motor.calcular_scores_questao(q_idx, questao)
        probs = motor.calibrar_probabilidades(s_hib)
        n_est, p_est, _ = motor.maximizar_probabilidade_arvore(probs)
        dec = {"decisao": "publicar", "tema_central_id": n_est}
        acertou = avaliar_acerto(dec, item, arvore)
        itens_eval.append({
            "id": qid,
            "p_est": p_est,
            "acertou": acertou
        })
        
    print(f"{'Limiar tau':<12} | {'Publicadas':<10} | {'Acertos':<8} | {'Precisão':<9} | {'Cobertura (N=100)':<18}")
    print("-" * 65)
    for tau in [0.85, 0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98]:
        pubs = [x for x in itens_eval if x["p_est"] >= tau]
        n_pub = len(pubs)
        n_ac = sum(1 for x in pubs if x["acertou"])
        prec = (n_ac / n_pub * 100) if n_pub > 0 else 0
        cob = n_pub / len(fv) * 100
        print(f"tau = {tau:.2f}     | {n_pub:<10} | {n_ac:<8} | {prec:6.2f}%   | {cob:5.1f}%")

if __name__ == "__main__":
    main()
