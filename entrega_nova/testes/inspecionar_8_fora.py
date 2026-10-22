"""Inspecionar os 8 itens fora_da_taxonomia na folha_verdade."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade

def main():
    questoes = carregar_e_processar_questoes()
    q_map = {q["id"]: q for q in questoes}
    fv = carregar_folha_verdade()
    
    for item in fv:
        if item["fora_da_taxonomia"] == 1:
            qid = item["id"]
            q = q_map[qid]
            print(f"=== {qid} (FORA DA TAXONOMIA) ===")
            print(f"Disciplina: {q.get('disciplina_origem')}")
            print(f"Assunto: {q.get('assunto_origem')}")
            print(f"Enunciado:\n{q['enunciado'][:250]}...")
            print()

if __name__ == "__main__":
    main()
