import json
import numpy as np
import sys
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao
from src.preparar import carregar_e_processar_questoes

motor = MotorDecisao()
questoes = carregar_e_processar_questoes()
q3 = [q for q in questoes if q['id'] == 'Q0003'][0]
print("Q0003 texto:\n", q3['texto_busca'])
print("Q0003 sinais:\n", q3['sinais_legais'])

q_idx = [i for i, q in enumerate(questoes) if q['id'] == 'Q0003'][0]
dense_scores = motor.dense_sim[q_idx]
top_idxs = np.argsort(dense_scores)[::-1][:10]
for idx in top_idxs:
    tid = motor.idx_to_tema[idx]
    t = motor.temas[tid]
    print(f"Dense Top: {dense_scores[idx]:.3f} | {tid} | {t['caminho']}")
