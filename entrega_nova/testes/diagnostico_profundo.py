"""Diagnóstico Profundo e Experimentos de Calibração, kNN e Análise da Taxonomia."""

import os
import sys
import json
import numpy as np
from collections import Counter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao
from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade

def main():
    motor = MotorDecisao()
    arvore = motor.arvore
    questoes = carregar_e_processar_questoes()
    fv = carregar_folha_verdade()
    fv_map = {item["id"]: item for item in fv}
    
    print("=== 1. ANÁLISE DA TAXONOMIA: NÍVEL 1 COM E SEM FILHOS ===")
    temas = arvore["temas"]
    filhos_map = arvore["filhos_map"]
    
    n1_com_filhos = 0
    n1_sem_filhos = 0
    for tid, info in temas.items():
        if info["nivel"] == 1:
            filhos = filhos_map.get(tid, [])
            if len(filhos) > 0:
                n1_com_filhos += 1
            else:
                n1_sem_filhos += 1
    print(f"Temas Nível 1 COM filhos: {n1_com_filhos}")
    print(f"Temas Nível 1 SEM filhos (folhas diretas): {n1_sem_filhos}")
    
    # Dentre os 53 rótulos de nível 1 na folha-verdade:
    n1_fv_com_filhos = 0
    n1_fv_sem_filhos = 0
    for item in fv:
        if item["fora_da_taxonomia"] == 0:
            tid = item["tema_central_id"]
            if temas.get(tid, {}).get("nivel") == 1:
                if len(filhos_map.get(tid, [])) > 0:
                    n1_fv_com_filhos += 1
                else:
                    n1_fv_sem_filhos += 1
    print(f"Folha-Verdade Nível 1 COM filhos: {n1_fv_com_filhos} ({n1_fv_com_filhos/53*100:.1f}%)")
    print(f"Folha-Verdade Nível 1 SEM filhos: {n1_fv_sem_filhos} ({n1_fv_sem_filhos/53*100:.1f}%)")
    
    print("\n=== 2. DESTINO DOS 8 ITENS FORA DA TAXONOMIA NA FOLHA-VERDADE ===")
    q_map = {q["id"]: (i, q) for i, q in enumerate(questoes)}
    destinos_fora = Counter()
    for item in fv:
        if item["fora_da_taxonomia"] == 1:
            qid = item["id"]
            idx, q = q_map[qid]
            dec = motor.decidir_questao(idx, q)
            destinos_fora[dec["decisao"]] += 1
            print(f"Questão {qid}: decisão = {dec['decisao']}, conf = {dec['confianca']:.2f}, motivo = {dec['motivo'][:40]}")
    print(f"Contagem destinos OOD: {destinos_fora}")

    print("\n=== 3. COBERTURA ESPECÍFICA NAS 559 QUESTÕES FORA DA FOLHA-VERDADE ===")
    pub_fv = 0
    rev_fv = 0
    fora_fv = 0
    
    pub_unlabeled = 0
    rev_unlabeled = 0
    fora_unlabeled = 0
    
    confs_pub = []
    
    for idx, q in enumerate(questoes):
        qid = q["id"]
        dec = motor.decidir_questao(idx, q)
        eh_fv = qid in fv_map
        
        if eh_fv:
            if dec["decisao"] == "publicar": pub_fv += 1
            elif dec["decisao"] == "revisar": rev_fv += 1
            else: fora_fv += 1
        else:
            if dec["decisao"] == "publicar":
                pub_unlabeled += 1
                confs_pub.append(dec["confianca"])
            elif dec["decisao"] == "revisar":
                rev_unlabeled += 1
            else:
                fora_unlabeled += 1
                
    total_unlabeled = len(questoes) - len(fv)
    print(f"Na Folha-Verdade (N=100): Publicar={pub_fv} ({pub_fv}%), Revisar={rev_fv}, Fora={fora_fv}")
    print(f"Nas Não-Rotuladas (N=559): Publicar={pub_unlabeled} ({pub_unlabeled/total_unlabeled*100:.2f}%), Revisar={rev_unlabeled}, Fora={fora_unlabeled}")
    print(f"Total (N=659): Publicar={pub_fv + pub_unlabeled} ({(pub_fv + pub_unlabeled)/659*100:.2f}%)")
    
    print("\n=== 4. VOLUMES REAIS POR FAIXA DE CONFIANÇA (FILA DE REVISÃO) ===")
    faixas = {
        "Nível 1 (Quase-Certeza: 0.80 <= P < 0.92)": 0,
        "Nível 2 (Disputa Dogmática: 0.50 <= P < 0.80)": 0,
        "Nível 3 (Baixa Aderência: P < 0.50)": 0,
    }
    
    for idx, q in enumerate(questoes):
        dec = motor.decidir_questao(idx, q)
        if dec["decisao"] == "revisar":
            conf = dec["confianca"]
            if conf >= 0.80:
                faixas["Nível 1 (Quase-Certeza: 0.80 <= P < 0.92)"] += 1
            elif conf >= 0.50:
                faixas["Nível 2 (Disputa Dogmática: 0.50 <= P < 0.80)"] += 1
            else:
                faixas["Nível 3 (Baixa Aderência: P < 0.50)"] += 1
                
    total_rev = sum(faixas.values())
    print(f"Total de questões em Revisar: {total_rev}")
    for k, v in faixas.items():
        pct = v / total_rev * 100
        proj_50k = int(round(pct / 100 * (50000 * (total_rev/659))))
        print(f"  {k}: {v} itens ({pct:.1f}% da fila) -> Projeção para 50k: ~{proj_50k} questões/semana")

if __name__ == "__main__":
    main()
