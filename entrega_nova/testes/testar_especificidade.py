import os
import json
import csv
import re
import numpy as np
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'artefatos', 'catalogo_arvore.json'), encoding='utf-8') as f:
    cat = json.load(f)

# Carrega questoes com texto_completo
questoes = []
with open(os.path.join(BASE_DIR, 'dados', 'questoes.jsonl'), encoding='utf-8') as f:
    for line in f:
        q = json.loads(line)
        enunciado = q.get('enunciado_html', '')
        comando = q.get('comando', '')
        alts = q.get('alternativas', {}) or {}
        partes = [enunciado, comando]
        if isinstance(alts, dict):
            partes.extend([f"({k}) {v}" for k, v in sorted(alts.items())])
        texto_comp = re.sub(r'<[^>]+>', ' ', "\n".join(partes))
        texto_comp = re.sub(r'\s+', ' ', texto_comp).strip()
        questoes.append({'id': q['id'], 'texto': texto_comp})

q_map = {q['id']: q['texto'] for q in questoes}

fv = []
with open(os.path.join(BASE_DIR, 'dados', 'folha_verdade.csv'), encoding='utf-8') as f:
    for r in csv.DictReader(f):
        fv.append(r)

print("Carregando intfloat/multilingual-e5-base...")
model = SentenceTransformer("intfloat/multilingual-e5-base")

temas_pub = [t for t in cat['temas'].values() if t['nivel'] in (1, 2) and t['tema_id'] not in ('5078', '5234')]
t_ids = [t['tema_id'] for t in temas_pub]

t_textos = []
for t in temas_pub:
    tit = t['titulo']
    cam = t['caminho']
    disc = t['disciplina']
    desc = t.get('descricao', '')
    desc_clean = re.sub(r'\d+\s*questões no source\.\s*Path:[^.]*\.?', '', desc, flags=re.IGNORECASE).strip()
    desc_clean = re.sub(r"Subtema de '[^']*'\s*\([^)]*\)\.?", '', desc_clean, flags=re.IGNORECASE).strip()
    
    texto = f"passage: Disciplina: {disc} | Ramo: {cam} | Tópico: {tit}."
    if desc_clean:
        texto += f" Definição: {desc_clean}"
    t_textos.append(texto)

print(f"Codificando {len(t_textos)} temas com e5-base...")
t_embs = model.encode(t_textos, batch_size=32, normalize_embeddings=True)

# Testa nas questões da folha-verdade
fv_val = [r for r in fv if r['fora_da_taxonomia'] != '1']
q_texts_val = [f"query: {q_map[r['id']]}" for r in fv_val]

print(f"Codificando {len(q_texts_val)} questões válidas da folha-verdade...")
q_embs_val = model.encode(q_texts_val, batch_size=32, normalize_embeddings=True)

sims_val = q_embs_val @ t_embs.T

acertos = 0
for i, r in enumerate(fv_val):
    true_tid = r['tema_central_id']
    sim_row = sims_val[i]
    top_idx = int(np.argmax(sim_row))
    pred_tid = t_ids[top_idx]
    
    pais = cat['pai_map'].get(true_tid)
    filhos = set(cat['filhos_map'].get(true_tid, []))
    ramo_verdade = {true_tid}
    if pais and cat['temas'][pais]['nivel'] >= 1:
        ramo_verdade.add(pais)
    ramo_verdade |= filhos
    
    if pred_tid in ramo_verdade:
        acertos += 1
    elif i < 15:
        print(f"Erro {r['id']}: Pred {pred_tid} ({cat['temas'][pred_tid]['caminho']}) | True {true_tid} ({r['tema_central_caminho']}) | Sim={sim_row[top_idx]:.3f}")

print(f"\n=======================================================")
print(f"RESULTADO EMPÍRICO COM intfloat/multilingual-e5-base:")
print(f"Acertos Diretos: {acertos}/{len(fv_val)} ({acertos/len(fv_val)*100:.1f}%)")
print(f"=======================================================")
