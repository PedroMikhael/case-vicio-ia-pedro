import json
import numpy as np
import sys
import os
from sentence_transformers import SentenceTransformer

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade

questoes = carregar_e_processar_questoes()
q3 = [q for q in questoes if q['id'] == 'Q0003'][0]

with open('entrega_nova/artefatos/catalogo_arvore.json', 'r', encoding='utf-8') as f:
    cat = json.load(f)

# Load model
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# Embed q3 with full text (enunciado + comando + alternativas)
q3_full_emb = model.encode([q3['texto_completo']], normalize_embeddings=True)[0]

# Compare against themes
with open('entrega_nova/artefatos/temas_enriquecidos.csv', 'r', encoding='utf-8') as f:
    import csv
    temas = list(csv.DictReader(f))

temas_pub = [t for t in temas if t['nivel'] in ('1', '2') and t['tema_id'] not in ('5078', '5234')]
t_texts = [t['texto_enriquecido'] for t in temas_pub]
t_embs = model.encode(t_texts, normalize_embeddings=True)

sims = t_embs @ q3_full_emb
top_idxs = np.argsort(sims)[::-1][:10]

print("=== TOP MATCHES COM TEXTO COMPLETO (COM ALTERNATIVAS) ===")
for idx in top_idxs:
    t = temas_pub[idx]
    print(f"Sim: {sims[idx]:.3f} | ID: {t['tema_id']} | {t['caminho']}")
