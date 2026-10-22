import numpy as np
from motor_decisao import MotorDecisao
from preparar import carregar_e_processar_questoes, carregar_folha_verdade
from validar_cv import avaliar_acerto

m = MotorDecisao()
qs = carregar_e_processar_questoes()
q_map = {q['id']: (i, q) for i, q in enumerate(qs)}
fv = carregar_folha_verdade()
arv = m.arvore

print("=== DIAGNOSTICO DE PREVISOES DA FOLHA-VERDADE ===")
for item in fv[:25]:
    idx, q = q_map[item['id']]
    s = m.calcular_scores_questao(idx, q)
    
    # Ver quais temas tiveram maior score bruto s
    top_idxs = np.argsort(s)[::-1][:3]
    top_themes = [(m.idx_to_tema[ti], s[ti], arv['temas'][m.idx_to_tema[ti]]['titulo']) for ti in top_idxs]
    
    p = m.calibrar_probabilidades(s, temperatura=0.03)
    n_est, p_est, _ = m.maximizar_probabilidade_arvore(p)
    dec = {'decisao': 'publicar', 'tema_central_id': n_est}
    ok = avaliar_acerto(dec, item, arv)
    
    t_true = item['tema_central_id']
    t_true_nome = arv['temas'].get(t_true, {}).get('titulo', 'FORA') if t_true else 'FORA'
    t_est_nome = arv['temas'].get(n_est, {}).get('titulo', '')
    
    print(f"ID: {item['id']} | FORA={item['fora_da_taxonomia']}")
    print(f"  Gabarito: {t_true} - {t_true_nome}")
    print(f"  n_estrela (max P): {n_est} - {t_est_nome} (P={p_est:.3f}) | ACERTO={ok}")
    print(f"  Top 3 scores brutos:")
    for tid, sc, tnome in top_themes:
        print(f"     {tid}: {sc:.3f} - {tnome}")
    print("-" * 50)
