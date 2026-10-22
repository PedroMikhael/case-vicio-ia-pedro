import csv
with open('entrega_nova/dados/temas.csv', encoding='utf-8') as f:
    for r in csv.DictReader(f):
        t = r['titulo'].lower()
        if any(w in t for w in ('recurso em sentido', 'rese', 'habeas corpus', 'furto', 'roubo', 'estelionato', 'homic')):
            print(f"{r['tema_id']} | nivel={r['nivel']} | {r['caminho']}")
