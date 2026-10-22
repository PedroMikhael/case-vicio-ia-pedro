import json
import numpy as np
from preparar import carregar_e_processar_questoes, carregar_folha_verdade
from validar_cv import avaliar_acerto

cat = json.load(open("entrega_nova/artefatos/catalogo_arvore.json", encoding="utf-8"))
cache = np.load("entrega_nova/artefatos/embeddings_cache.npz")
questoes = carregar_e_processar_questoes()
q_map = {q["id"]: (i, q) for i, q in enumerate(questoes)}
fv = carregar_folha_verdade()

nos_pub = cat["nos_publicaveis"]
tema_idx_map = {tid: i for i, tid in enumerate(nos_pub)}
idx_to_tema = {i: tid for i, tid in enumerate(nos_pub)}
n_temas = len(nos_pub)

A_mat = np.zeros((n_temas, n_temas), dtype=np.float32)
for i, n in enumerate(nos_pub):
    for t in cat["dominio_acerto"].get(n, [n]):
        if t in tema_idx_map:
            A_mat[i, tema_idx_map[t]] = 1.0

dense_sim = cache["questoes_dense"] @ cache["temas_dense"].T
sparse_sim = cache["sparse_sim"]

# Investigação detalhada de cada questão da folha-verdade:
# Qual é o tema predito pelo dense_sim puro?
acertos_dense = 0
print("=== AVALIACAO DO DENSE SIM PURO (TOP 1 DIRETO) ===")
for item in fv:
    if item["fora_da_taxonomia"] == 1:
        continue
    idx, q = q_map[item["id"]]
    top_t_idx = int(np.argmax(dense_sim[idx]))
    top_tid = idx_to_tema[top_t_idx]
    dec = {"decisao": "publicar", "tema_central_id": top_tid}
    ok = avaliar_acerto(dec, item, cat)
    if ok:
        acertos_dense += 1

print(f"Top 1 Dense Puro (sem filtros): {acertos_dense} / 92 ({acertos_dense/92*100:.1f}%)")

# Qual é o acerto pelo sparse_sim puro?
acertos_sparse = 0
for item in fv:
    if item["fora_da_taxonomia"] == 1:
        continue
    idx, q = q_map[item["id"]]
    top_t_idx = int(np.argmax(sparse_sim[idx]))
    top_tid = idx_to_tema[top_t_idx]
    dec = {"decisao": "publicar", "tema_central_id": top_tid}
    ok = avaliar_acerto(dec, item, cat)
    if ok:
        acertos_sparse += 1

print(f"Top 1 Sparse Puro (sem filtros): {acertos_sparse} / 92 ({acertos_sparse/92*100:.1f}%)")

# E se combinarmos A_mat com argmax direto (sem softmax)?
# score_ramo = A_mat @ dense_sim[idx]
acertos_ramo = 0
for item in fv:
    if item["fora_da_taxonomia"] == 1:
        continue
    idx, q = q_map[item["id"]]
    ramo_scores = A_mat @ dense_sim[idx]
    top_t_idx = int(np.argmax(ramo_scores))
    top_tid = idx_to_tema[top_t_idx]
    dec = {"decisao": "publicar", "tema_central_id": top_tid}
    ok = avaliar_acerto(dec, item, cat)
    if ok:
        acertos_ramo += 1

print(f"Top 1 Ramo Acumulado A_mat @ dense: {acertos_ramo} / 92 ({acertos_ramo/92*100:.1f}%)")
