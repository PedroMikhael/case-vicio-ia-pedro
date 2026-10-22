"""Inspecionar Q0424, Q0527, Q0649."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.preparar import carregar_e_processar_questoes

def main():
    questoes = carregar_e_processar_questoes()
    q_map = {q["id"]: q for q in questoes}
    for qid in ["Q0424", "Q0527", "Q0649"]:
        q = q_map[qid]
        print(f"=== {qid} ===")
        print(f"Disciplina: {q.get('disciplina_origem')}")
        print(f"Assunto: {q.get('assunto_origem')}")
        print(f"Enunciado:\n{q['enunciado'][:300]}")
        print()

if __name__ == "__main__":
    main()
