"""Inspeção detalhada de Q0313."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao
from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade

def main():
    motor = MotorDecisao()
    arvore = motor.arvore
    questoes = carregar_e_processar_questoes()
    fv = carregar_folha_verdade()
    
    q_dict = {q["id"]: (i, q) for i, q in enumerate(questoes)}
    idx, q = q_dict["Q0313"]
    
    print("=== DADOS DA QUESTÃO Q0313 ===")
    print(f"Disciplina Origem: {q.get('disciplina_origem')}")
    print(f"Assunto Origem: {q.get('assunto_origem')}")
    print(f"Enunciado:\n{q['enunciado']}")
    print(f"\nAlternativas:\n{q.get('alternativas')}")
    
    s = motor.calcular_scores_questao(idx, q)
    probs = motor.calibrar_probabilidades(s)
    
    # Top 5 temas por probabilidade individual
    top_indices = np.argsort(probs)[::-1][:5]
    print("\nTop 5 temas individuais:")
    for rank, ti in enumerate(top_indices, 1):
        tid = motor.idx_to_tema[ti]
        tinfo = arvore["temas"].get(tid, {})
        print(f"  {rank}. ID {tid}: {tinfo.get('titulo')} ({tinfo.get('caminho')}) -> prob={probs[ti]:.4f}, score={s[ti]:.4f}")
        
    n_est, p_est, _ = motor.maximizar_probabilidade_arvore(probs)
    print(f"\nDecisão da árvore: {n_est} ({arvore['temas'].get(n_est, {}).get('titulo')}) com P(acerto)={p_est:.4f}")

if __name__ == "__main__":
    import numpy as np
    main()
