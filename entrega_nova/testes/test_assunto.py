import json
import csv
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, 'artefatos', 'catalogo_arvore.json'), encoding='utf-8') as f:
    cat = json.load(f)

fv = {}
with open(os.path.join(BASE_DIR, 'dados', 'folha_verdade.csv'), encoding='utf-8') as f:
    for r in csv.DictReader(f):
        fv[r['id']] = r

questoes = {}
with open(os.path.join(BASE_DIR, 'dados', 'questoes.jsonl'), encoding='utf-8') as f:
    for line in f:
        q = json.loads(line)
        questoes[q['id']] = q

matched = 0
total = 0

print("=== COMPARAÇÃO ASSUNTO_ORIGEM vs FOLHA-VERDADE ===")
for qid in sorted(fv.keys())[:30]:
    fv_row = fv[qid]
    if fv_row['fora_da_taxonomia'] == '1':
        print(f"{qid} | [FORA DA TAXONOMIA] | Origem: {questoes[qid].get('assunto_origem')} ({questoes[qid].get('disciplina_origem')})")
        continue
    total += 1
    q = questoes[qid]
    assunto = q.get('assunto_origem', '')
    disc = q.get('disciplina_origem', '')
    true_tid = fv_row['tema_central_id']
    true_cam = fv_row['tema_central_caminho']
    
    # Verifica acerto do ramo
    pais = cat['pai_map'].get(true_tid)
    filhos = set(cat['filhos_map'].get(true_tid, []))
    ramo_verdade = {true_tid}
    if pais and cat['temas'][pais]['nivel'] >= 1:
        ramo_verdade.add(pais)
    ramo_verdade |= filhos
    
    # Encontra melhor match do título do tema no assunto
    melhor_cand = None
    melhor_score = 0
    for tid, t in cat['temas'].items():
        if t['nivel'] == 0:
            continue
        tit = t['titulo'].lower()
        ass = assunto.lower()
        if tit == ass:
            melhor_cand = tid
            break
        elif tit in ass or ass in tit:
            if len(tit) > melhor_score:
                melhor_score = len(tit)
                melhor_cand = tid
                
    acertou = (melhor_cand in ramo_verdade) if melhor_cand else False
    if acertou:
        matched += 1
    cand_tit = cat['temas'][melhor_cand]['titulo'] if melhor_cand else 'None'
    print(f"{qid} | Acertou={acertou} | Assunto='{assunto}' -> Cand='{cand_tit}' | True='{true_cam}'")

print(f"\nTaxa de correspondência direta simples por título: {matched}/{total}")
