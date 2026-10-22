import json
import numpy as np
import sys
import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

cache = np.load(os.path.join(BASE_DIR, 'artefatos', 'embeddings_cache.npz'))
q_dense = cache['questoes_dense']
t_dense = cache['temas_dense']
q_ids = list(cache['questoes_ids'])
t_ids = list(cache['temas_ids'])

with open(os.path.join(BASE_DIR, 'artefatos', 'catalogo_arvore.json'), encoding='utf-8') as f:
    cat = json.load(f)

fv = []
with open(os.path.join(BASE_DIR, 'dados', 'folha_verdade.csv'), encoding='utf-8') as f:
    for r in csv.DictReader(f):
        fv.append(r)

sims = q_dense @ t_dense.T  # (659, 371)

acertos_raw = 0
total_avaliado = 0

print("=== VERIFICAÇÃO DE MATCHES DIRETOS DO COSSENO DENSO ===")
for r in fv:
    if r['fora_da_taxonomia'] == '1':
        continue
    total_avaliado += 1
    qid = r['id']
    q_idx = q_ids.index(qid)
    q_sim = sims[q_idx]
    top_t_idx = int(np.argmax(q_sim))
    top_tid = t_ids[top_t_idx]
    true_tid = r['tema_central_id']
    
    # Verifica se top_tid acerta pela regra (ele mesmo, pai ou filho)
    pais = cat['pai_map'].get(true_tid)
    filhos = set(cat['filhos_map'].get(true_tid, []))
    ramo_verdade = {true_tid}
    if pais and cat['temas'][pais]['nivel'] >= 1:
        ramo_verdade.add(pais)
    ramo_verdade |= filhos
    
    acertou = top_tid in ramo_verdade
    if acertou:
        acertos_raw += 1
        
    pred_caminho = cat['temas'][top_tid]['caminho']
    true_caminho = r['tema_central_caminho']
    if total_avaliado <= 20:
        print(f"{qid} | Acertou={acertou} | Sim={q_sim[top_t_idx]:.3f}")
        print(f"   PRED: {top_tid} -> {pred_caminho}")
        print(f"   TRUE: {true_tid} -> {true_caminho}")

print(f"\nTotal Acertos Diretos do Cosseno Denso: {acertos_raw}/{total_avaliado} ({acertos_raw/total_avaliado*100:.1f}%)")
