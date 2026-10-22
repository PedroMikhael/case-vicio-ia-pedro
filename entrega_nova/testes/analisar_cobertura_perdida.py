"""Análise detalhada da cobertura perdida (falsos negativos na fila de revisão)."""

import os
import sys

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
    
    corretos_revisar = []
    
    for item in fv:
        qid = item["id"]
        q_idx, q = q_map[qid]
        s = motor.calcular_scores_questao(q_idx, q)
        probs = motor.calibrar_probabilidades(s)
        n_est, p_est, _ = motor.maximizar_probabilidade_arvore(probs)
        
        dec = {"decisao": "publicar", "tema_central_id": n_est}
        acertou = avaliar_acerto(dec, item, arvore)
        
        # Foi para revisar pelo motor?
        dec_real = motor.decidir_questao(q_idx, q)
        if dec_real["decisao"] == "revisar" and acertou:
            corretos_revisar.append({
                "id": qid,
                "n_est": n_est,
                "titulo": arvore["temas"].get(n_est, {}).get("titulo"),
                "p_est": p_est,
                "len_texto": len(q["texto_completo"]),
                "enunciado": q["enunciado"][:100]
            })
            
    print(f"Total de itens na folha-verdade CORRETOS no Top-1 que foram para REVISAR: {len(corretos_revisar)}")
    print("\nExemplos de Cobertura Perdida (Acerto com P < 0.92):")
    for r in sorted(corretos_revisar, key=lambda x: x["p_est"], reverse=True)[:10]:
        print(f"  {r['id']} (P = {r['p_est']:.4f}, len={r['len_texto']}): {r['n_est']} ({r['titulo']}) | {r['enunciado']}...")
        
    # Análise de propriedades dos corretos em revisar
    p_vals = [r["p_est"] for r in corretos_revisar]
    len_vals = [r["len_texto"] for r in corretos_revisar]
    import numpy as np
    print(f"\nMédia de P(acerto) dos corretos em revisar: {np.mean(p_vals):.4f} (mediana: {np.median(p_vals):.4f})")
    print(f"Média de comprimento dos corretos em revisar: {np.mean(len_vals):.1f} caracteres")

if __name__ == "__main__":
    main()
