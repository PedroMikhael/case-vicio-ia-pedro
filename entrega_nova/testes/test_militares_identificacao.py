"""Teste de correção do P(fora) para questões militares e temas inativos."""

import os
import sys
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao
from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade

def main():
    motor = MotorDecisao()
    questoes = carregar_e_processar_questoes()
    fv = carregar_folha_verdade()
    fv_map = {item["id"]: item for item in fv}
    
    # Identificar quais questões das 659 são militares
    termos_militares = [
        "militar", "cpm", "cppm", "código penal militar", "processo penal militar",
        "inquérito policial militar", "ipm", "superior tribunal militar", "stm",
        "justiça militar"
    ]
    
    militares_ids = set()
    for q in questoes:
        disc = (q.get("disciplina_origem") or "").lower()
        assunto = (q.get("assunto_origem") or "").lower()
        enunc = q["enunciado"].lower()
        
        # Critério estrito de matéria militar (escopo da jurisdição / código especial)
        if "militar" in disc or "militar" in assunto:
            militares_ids.add(q["id"])
            
    print(f"Total de questões identificadas com origem Militar: {len(militares_ids)}")
    print(f"IDs: {sorted(list(militares_ids))}")
    
    # Verificar se alguma está na folha-verdade
    mil_na_fv = [qid for qid in militares_ids if qid in fv_map]
    print(f"Militares na folha-verdade: {len(mil_na_fv)} ({mil_na_fv})")

if __name__ == "__main__":
    main()
