import os
import json
import csv
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'artefatos', 'catalogo_arvore.json'), encoding='utf-8') as f:
    cat = json.load(f)

fv = []
with open(os.path.join(BASE_DIR, 'dados', 'folha_verdade.csv'), encoding='utf-8') as f:
    for r in csv.DictReader(f):
        fv.append(r)

cache = np.load(os.path.join(BASE_DIR, 'artefatos', 'embeddings_cache.npz'))
q_dense = cache['questoes_dense']
t_dense = cache['temas_dense']
q_ids = list(cache['questoes_ids'])
t_ids = list(cache['temas_ids'])

sims = q_dense @ t_dense.T # (659, 371)

# Verifica para cada questão da folha-verdade:
# - melhor score s1
# - segundo melhor score s2
# - margem delta = s1 - s2
# - se s1 acertou

dados_fv = []
for r in fv:
    qid = r['id']
    q_idx = q_ids.index(qid)
    row_sims = sims[q_idx]
    
    sorted_idxs = np.argsort(row_sims)[::-1]
    top1_idx = sorted_idxs[0]
    top2_idx = sorted_idxs[1]
    
    s1 = float(row_sims[top1_idx])
    s2 = float(row_sims[top2_idx])
    delta = s1 - s2
    
    top1_tid = t_ids[top1_idx]
    eh_fora = (r['fora_da_taxonomia'] == '1')
    true_tid = r['tema_central_id']
    
    if eh_fora:
        acertou = False
    else:
        pais = cat['pai_map'].get(true_tid)
        filhos = set(cat['filhos_map'].get(true_tid, []))
        ramo_verdade = {true_tid}
        if pais and cat['temas'][pais]['nivel'] >= 1:
            ramo_verdade.add(pais)
        ramo_verdade |= filhos
        acertou = (top1_tid in ramo_verdade)
        
    dados_fv.append({
        'id': qid,
        'eh_fora': eh_fora,
        's1': s1,
        's2': s2,
        'delta': delta,
        'acertou': acertou,
        'top1_tid': top1_tid,
        'true_tid': true_tid
    })

print("=== DISTRIBUIÇÃO DE MARGENS E PRECISÃO POR FAIXA ===")
for th_delta in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
    filtrados = [d for d in dados_fv if d['delta'] >= th_delta and d['s1'] >= 0.60 and not d['eh_fora']]
    n_pub = len(filtrados)
    n_acertos = sum(1 for d in filtrados if d['acertou'])
    prec = (n_acertos / n_pub * 100.0) if n_pub > 0 else 0.0
    print(f"Delta >= {th_delta:.2f} & S1 >= 0.60: {n_acertos}/{n_pub} ({prec:.1f}% precisão, {n_pub/len(fv)*100:.1f}% cobertura)")
