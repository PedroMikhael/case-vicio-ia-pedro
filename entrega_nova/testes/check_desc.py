import csv

with open('entrega_nova/dados/temas.csv', encoding='utf-8') as f:
    temas = list(csv.DictReader(f))

tem_desc_rica = 0
for t in temas:
    desc = t.get('descricao', '').strip()
    if desc and not desc.startswith('Disciplina (root)') and 'questões no source' not in desc:
        tem_desc_rica += 1

print(f"Total de temas com descrição jurídica rica pré-existente: {tem_desc_rica}/{len(temas)}")
