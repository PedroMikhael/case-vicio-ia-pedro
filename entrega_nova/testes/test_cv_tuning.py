import os
import json
import csv
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'artefatos', 'catalogo_arvore.json'), encoding='utf-8') as f:
    cat = json.load(f)

fv = []
with open(os.path.join(BASE_DIR, 'dados', 'folha_verdade.csv'), encoding='utf-8') as f:
    for r in csv.DictReader(f):
        fv.append(r)

# Carrega questoes
questoes_dict = {}
with open(os.path.join(BASE_DIR, 'dados', 'questoes.jsonl'), encoding='utf-8') as f:
    for line in f:
        q = json.loads(line)
        questoes_dict[q['id']] = q

# Model
model = SentenceTransformer("intfloat/multilingual-e5-base")

temas_pub = [t for t in cat['temas'].values() if t['nivel'] in (1, 2) and t['tema_id'] not in ('5078', '5234')]
t_ids = [t['tema_id'] for t in temas_pub]
t_idx_map = {tid: i for i, tid in enumerate(t_ids)}

# Enriched theme texts
with open(os.path.join(BASE_DIR, 'artefatos', 'temas_enriquecidos.csv'), encoding='utf-8') as f:
    temas_enriq = {r['tema_id']: r['texto_enriquecido'] for r in csv.DictReader(f)}

t_texts = [f"passage: {temas_enriq[tid]}" for tid in t_ids]
t_embs = model.encode(t_texts, batch_size=32, normalize_embeddings=True)

# Question texts for 100 folha-verdade
q_texts = []
for r in fv:
    qid = r['id']
    q = questoes_dict[qid]
    enunc = q.get('enunciado_html', '')
    cmd = q.get('comando', '')
    alts = q.get('alternativas', {}) or {}
    partes = [enunc, cmd]
    if isinstance(alts, dict):
        partes.extend([f"({k}) {v}" for k, v in sorted(alts.items())])
    import re
    t_comp = re.sub(r'<[^>]+>', ' ', "\n".join(partes))
    t_comp = re.sub(r'\s+', ' ', t_comp).strip()
    q_texts.append(f"query: {t_comp}")

q_embs = model.encode(q_texts, batch_size=32, normalize_embeddings=True)

# Dot product questions x themes: (100, 371)
sims_qt = q_embs @ t_embs.T

# Dot product questions x questions: (100, 100)
sims_qq = q_embs @ q_embs.T

print("=== TESTANDO VALIDAÇÃO CRUZADA 5-FOLD COM CALIBRAÇÃO DE LIMIAR ===")

estratos = [("FORA" if r['fora_da_taxonomia'] == '1' else cat['temas'].get(r['tema_central_id'], {}).get('disciplina', 'OUTRA')) for r in fv]

for th_margin in [0.01, 0.02, 0.03, 0.04, 0.05]:
    for th_score in [0.82, 0.84, 0.86, 0.88]:
        total_pub = 0
        total_acerto = 0
        total_fora_correto = 0
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for train_idx, test_idx in skf.split(range(len(fv)), estratos):
            # Exemplar prototypes from train_idx
            train_items = [fv[i] for i in train_idx if fv[i]['fora_da_taxonomia'] != '1']
            train_embs = q_embs[[i for i in train_idx if fv[i]['fora_da_taxonomia'] != '1']]
            train_tids = [r['tema_central_id'] for r in train_items]
            
            for i in test_idx:
                r = fv[i]
                eh_fora = (r['fora_da_taxonomia'] == '1')
                true_tid = r['tema_central_id']
                
                # Similaridade direta com temas
                s_temas = sims_qt[i].copy()
                
                # Bonus k-NN dos 80 exemplos de treino
                if len(train_embs) > 0:
                    sim_to_train = q_embs[i] @ train_embs.T # (n_train,)
                    for tr_idx, sim_val in enumerate(sim_to_train):
                        if sim_val > 0.85:
                            tr_tid = train_tids[tr_idx]
                            if tr_tid in t_idx_map:
                                s_temas[t_idx_map[tr_tid]] += (sim_val - 0.80) * 0.5
                                
                top_idxs = np.argsort(s_temas)[::-1]
                s1 = s_temas[top_idxs[0]]
                s2 = s_temas[top_idxs[1]]
                delta = s1 - s2
                pred_tid = t_ids[top_idxs[0]]
                
                # Regra de decisão
                if s1 >= th_score and delta >= th_margin:
                    total_pub += 1
                    if not eh_fora:
                        pais = cat['pai_map'].get(true_tid)
                        filhos = set(cat['filhos_map'].get(true_tid, []))
                        ramo_verdade = {true_tid}
                        if pais and cat['temas'][pais]['nivel'] >= 1:
                            ramo_verdade.add(pais)
                        ramo_verdade |= filhos
                        if pred_tid in ramo_verdade:
                            total_acerto += 1
                            
        prec = (total_acerto / total_pub * 100.0) if total_pub > 0 else 0.0
        cob = (total_pub / len(fv) * 100.0)
        if prec >= 90.0 and cob >= 15.0:
            print(f"Limiar Score >= {th_score:.2f} | Margem >= {th_margin:.2f} -> Precisão: {total_acerto}/{total_pub} ({prec:.1f}%) | Cobertura: {cob:.1f}%")
