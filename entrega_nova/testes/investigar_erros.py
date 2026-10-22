"""Investigar os erros em cada ponto da curva."""

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
    
    resultados = []
    for item in fv:
        qid = item["id"]
        q_idx, questao = q_map[qid]
        s_hib = motor.calcular_scores_questao(q_idx, questao)
        probs = motor.calibrar_probabilidades(s_hib)
        n_est, p_est, p_acum = motor.maximizar_probabilidade_arvore(probs)
        dec = {"decisao": "publicar", "tema_central_id": n_est}
        acertou = avaliar_acerto(dec, item, arvore)
        resultados.append({
            "id": qid,
            "n_est": n_est,
            "nome_est": arvore["temas"].get(n_est, {}).get("nome", ""),
            "p_est": p_est,
            "verdade": item["tema_central_id"],
            "nome_verdade": arvore["temas"].get(item["tema_central_id"], {}).get("nome", "") if item["fora_da_taxonomia"] == 0 else "FORA",
            "acertou": acertou,
            "fora": item["fora_da_taxonomia"]
        })
        
    resultados_ord = sorted(resultados, key=lambda x: x["p_est"], reverse=True)
    
    print("ERROS ENTRE OS PRIMEIROS 30 ITENS:")
    for rank, r in enumerate(resultados_ord[:30], start=1):
        if not r["acertou"]:
            print(f"Rank {rank:2d}: {r['id']} (P = {r['p_est']:.4f}) -> Predito: {r['n_est']} ({r['nome_est']}) | Verdade: {r['verdade']} ({r['nome_verdade']}) | Fora: {r['fora']}")

if __name__ == "__main__":
    main()
