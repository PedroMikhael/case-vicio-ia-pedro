"""Teste de detecção OOD corrigindo o limiar em estimar_probabilidade_fora."""

import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao, DISCIPLINAS_ESTRANHAS
from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade

def main():
    motor = MotorDecisao()
    questoes = carregar_e_processar_questoes()
    fv = carregar_folha_verdade()
    fv_map = {item["id"]: item for item in fv}
    
    fora_detectados = []
    
    for idx, q in enumerate(questoes):
        disc_orig = (q.get("disciplina_origem") or "").lower().strip()
        assunto_orig = (q.get("assunto_origem") or "").lower().strip()
        
        eh_estranha = any(d in disc_orig or d in assunto_orig for d in DISCIPLINAS_ESTRANHAS)
        if eh_estranha:
            fora_detectados.append((q["id"], disc_orig, assunto_orig, q["id"] in fv_map))
            
    print(f"Total de questões com disciplina/assunto estranho: {len(fora_detectados)}")
    for qid, disc, ass, eh_fv in fora_detectados:
        print(f"  {qid} (Na Folha? {eh_fv}): disc='{disc}', ass='{ass}'")

if __name__ == "__main__":
    main()
