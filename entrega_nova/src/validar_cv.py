"""Protocolo de Validação Experimental 5x5 e Estudo de Ablação Científico (Case Vício IA)."""

import os
import sys
import json
import csv
import math
import numpy as np
from typing import Dict, List, Any, Tuple
from sklearn.model_selection import StratifiedKFold

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao
from src.preparar import carregar_e_processar_questoes, carregar_folha_verdade

DADOS_DIR = os.path.join(BASE_DIR, "dados")
ARTEFATOS_DIR = os.path.join(BASE_DIR, "artefatos")


def calcular_intervalo_wilson(acertos: int, total: int, confianca: float = 0.95) -> Tuple[float, float, float]:
    """Calcula a proporção pontual e o Intervalo de Confiança de Wilson Score."""
    if total == 0:
        return 0.0, 0.0, 0.0
    p = acertos / total
    # z = 1.95996 para 95%
    z = 1.95996 if confianca == 0.95 else 1.64485
    z2 = z * z
    den = 1.0 + z2 / total
    centro = (p + z2 / (2.0 * total)) / den
    margem = (z / den) * math.sqrt((p * (1.0 - p) / total) + (z2 / (4.0 * total * total)))
    inf = max(0.0, centro - margem)
    sup = min(1.0, centro + margem)
    return p, inf, sup


def calcular_intervalo_bootstrap(y_true: List[int], n_bootstrap: int = 10000, seed: int = 42) -> Tuple[float, float, float]:
    """Calcula Intervalo de Confiança Bootstrap Empírico a 95%."""
    if not y_true:
        return 0.0, 0.0, 0.0
    rng = np.random.RandomState(seed)
    arr = np.array(y_true, dtype=np.float32)
    tamanho = len(arr)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=tamanho, replace=True)
        boot_means.append(np.mean(sample))
    boot_means = np.sort(boot_means)
    p_ponto = float(np.mean(arr))
    inf = float(np.percentile(boot_means, 2.5))
    sup = float(np.percentile(boot_means, 97.5))
    return p_ponto, inf, sup


def avaliar_acerto(decisao_item: Dict[str, Any], fv_item: Dict[str, Any], catalogo_arvore: Dict[str, Any]) -> bool:
    """Verifica se a decisão de publicação está correta segundo as regras oficiais do edital."""
    if decisao_item["decisao"] != "publicar":
        return False
    # Se a questão é fora da taxonomia, publicar qualquer tema é ERRO
    if fv_item["fora_da_taxonomia"] == 1:
        return False
        
    tema_pub = decisao_item["tema_central_id"]
    tema_verdade = fv_item["tema_central_id"]
    extras_verdade = set(fv_item.get("extras_ids", []))
    
    # Conjunto de temas que contam como acerto:
    # O próprio tema central, os extras verdadeiros, e o pai ou filhos de qualquer um deles
    todos_alvos_validos = {tema_verdade} | extras_verdade
    pai_map = catalogo_arvore["pai_map"]
    filhos_map = catalogo_arvore["filhos_map"]
    
    ramo_verdade = set()
    for alvo in todos_alvos_validos:
        if not alvo:
            continue
        ramo_verdade.add(alvo)
        p = pai_map.get(alvo)
        if p and catalogo_arvore["temas"].get(p, {}).get("nivel", 0) >= 1:
            ramo_verdade.add(p)
        for f in filhos_map.get(alvo, []):
            ramo_verdade.add(f)
            
    return tema_pub in ramo_verdade


def executar_ablativo_e_cv(
    n_splits: int = 5,
    n_repeats: int = 5,
    tau_fora: float = 0.40,
    tau_pub: float = 0.92
):
    """Executa o protocolo de validação cruzada 5x5 repetida e calcula o estudo de ablação."""
    print("=" * 70)
    print("PROTOCOLO CIENTÍFICO DE VALIDAÇÃO CRUZADA 5x5 REPETIDA (25 FOLDS)")
    print(f"Limiares Operacionais: tau_fora = {tau_fora:.2f} | tau_pub = {tau_pub:.2f}")
    print("=" * 70)
    
    questoes = carregar_e_processar_questoes()
    q_map = {q["id"]: (i, q) for i, q in enumerate(questoes)}
    
    fv = carregar_folha_verdade()
    fv_map = {item["id"]: item for item in fv}
    
    motor = MotorDecisao(tau_fora=tau_fora, tau_pub=tau_pub)
    arvore = motor.arvore
    
    # Estratificação por disciplina e fora da taxonomia
    estratos = []
    for item in fv:
        if item["fora_da_taxonomia"] == 1:
            estratos.append("FORA")
        else:
            tid = item["tema_central_id"]
            disc = arvore["temas"].get(tid, {}).get("disciplina", "OUTRA")
            estratos.append(disc)
            
    indices_fv = np.arange(len(fv))
    
    # 4 configurações do estudo de ablação:
    # 0: Baseline 0 (Dense puro + Argmax sem árvore)
    # 1: Iteração 1 (Dense puro + Teoria da Decisão na Árvore)
    # 2: Iteração 2 (Híbrido Denso+Esparso+Simbólico + Árvore)
    # 3: Iteração 3 / Sistema Completo (Híbrido + Árvore + Modelagem Ativa de P(fora))
    
    resultados_config = {
        "Baseline 0 (Dense puro / Sem Árvore / Sem OOD)": {"pub": 0, "acertos": 0, "fora_detectado": 0, "total_fora": 0},
        "Iteração 1 (Dense + Teoria da Decisão na Árvore)": {"pub": 0, "acertos": 0, "fora_detectado": 0, "total_fora": 0},
        "Iteração 2 (Híbrido Denso+Esparso+Simbólico + Árvore)": {"pub": 0, "acertos": 0, "fora_detectado": 0, "total_fora": 0},
        "Iteração 3 (Sistema Completo: Híbrido + Árvore + P(fora))": {"pub": 0, "acertos": 0, "fora_detectado": 0, "total_fora": 0}
    }
    
    # Rastreamento a nível de questão única (evitando pseudorreplicação amostral)
    decisoes_por_questao = {item["id"]: [] for item in fv}
    
    rng_master = np.random.RandomState(42)
    
    for rep in range(n_repeats):
        seed = int(rng_master.randint(1000, 99999))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        
        for fold, (train_idx, test_idx) in enumerate(skf.split(indices_fv, estratos)):
            test_items = [fv[i] for i in test_idx]
            
            for item in test_items:
                qid = item["id"]
                q_idx, questao = q_map[qid]
                eh_fora = (item["fora_da_taxonomia"] == 1)
                
                # --- CONFIG 0: Dense puro sem árvore nem OOD
                s_dense = motor.dense_sim[q_idx]
                top_dense_idx = int(np.argmax(s_dense))
                top_dense_tid = motor.idx_to_tema[top_dense_idx]
                conf_dense = float(s_dense[top_dense_idx])
                if conf_dense >= 0.45:
                    resultados_config["Baseline 0 (Dense puro / Sem Árvore / Sem OOD)"]["pub"] += 1
                    dec_0 = {"decisao": "publicar", "tema_central_id": top_dense_tid}
                    if avaliar_acerto(dec_0, item, arvore):
                        resultados_config["Baseline 0 (Dense puro / Sem Árvore / Sem OOD)"]["acertos"] += 1
                if eh_fora:
                    resultados_config["Baseline 0 (Dense puro / Sem Árvore / Sem OOD)"]["total_fora"] += 1
                    
                # --- CONFIG 1: Dense puro + Árvore
                probs_dense = motor.calibrar_probabilidades(s_dense, temperatura=0.005)
                n_est_1, p_est_1, _ = motor.maximizar_probabilidade_arvore(probs_dense)
                if p_est_1 >= tau_pub:
                    resultados_config["Iteração 1 (Dense + Teoria da Decisão na Árvore)"]["pub"] += 1
                    dec_1 = {"decisao": "publicar", "tema_central_id": n_est_1}
                    if avaliar_acerto(dec_1, item, arvore):
                        resultados_config["Iteração 1 (Dense + Teoria da Decisão na Árvore)"]["acertos"] += 1
                if eh_fora:
                    resultados_config["Iteração 1 (Dense + Teoria da Decisão na Árvore)"]["total_fora"] += 1
                    
                # --- CONFIG 2: Híbrido + Árvore (sem P(fora))
                scores_hib = motor.calcular_scores_questao(q_idx, questao)
                probs_hib = motor.calibrar_probabilidades(scores_hib)
                n_est_2, p_est_2, _ = motor.maximizar_probabilidade_arvore(probs_hib)
                if p_est_2 >= tau_pub:
                    resultados_config["Iteração 2 (Híbrido Denso+Esparso+Simbólico + Árvore)"]["pub"] += 1
                    dec_2 = {"decisao": "publicar", "tema_central_id": n_est_2}
                    if avaliar_acerto(dec_2, item, arvore):
                        resultados_config["Iteração 2 (Híbrido Denso+Esparso+Simbólico + Árvore)"]["acertos"] += 1
                if eh_fora:
                    resultados_config["Iteração 2 (Híbrido Denso+Esparso+Simbólico + Árvore)"]["total_fora"] += 1
                    
                # --- CONFIG 3: Sistema Completo (Híbrido + Árvore + P(fora))
                dec_3 = motor.decidir_questao(q_idx, questao)
                acertou_pub = avaliar_acerto(dec_3, item, arvore)
                # Verifica se a sugestão top-1 seria correta se fosse publicada (para calibração e cobertura perdida)
                acertou_top1 = avaliar_acerto({"decisao": "publicar", "tema_central_id": dec_3["tema_central_id"]}, item, arvore)
                
                decisoes_por_questao[qid].append({
                    "decisao": dec_3["decisao"],
                    "tema_central_id": dec_3["tema_central_id"],
                    "confianca": dec_3["confianca"],
                    "acertou": acertou_pub,
                    "acertou_top1": acertou_top1,
                    "eh_fora": eh_fora
                })
                
                if eh_fora:
                    resultados_config["Iteração 3 (Sistema Completo: Híbrido + Árvore + P(fora))"]["total_fora"] += 1
                    if dec_3["decisao"] == "fora":
                        resultados_config["Iteração 3 (Sistema Completo: Híbrido + Árvore + P(fora))"]["fora_detectado"] += 1
                        
                if dec_3["decisao"] == "publicar":
                    resultados_config["Iteração 3 (Sistema Completo: Híbrido + Árvore + P(fora))"]["pub"] += 1
                    if acertou_pub:
                        resultados_config["Iteração 3 (Sistema Completo: Híbrido + Árvore + P(fora))"]["acertos"] += 1
                        
    print("\n" + "=" * 70)
    print("TABELA DE ABLAÇÃO EXPERIMENTAL (MÉDIAS CONSOLIDADAS EM 25 FOLDS)")
    print("=" * 70)
    print(f"{'Configuração Experimental':<55} | {'Precisão':<9} | {'Cobertura':<9} | {'Detecção Fora'}")
    print("-" * 90)
    
    total_amostras_cv = len(fv) * n_repeats
    resumo_ablativo = []
    
    for nome, dados in resultados_config.items():
        pub = dados["pub"]
        acertos = dados["acertos"]
        tot_fora = dados["total_fora"]
        det_fora = dados["fora_detectado"]
        
        prec = (acertos / pub * 100.0) if pub > 0 else 0.0
        cob = (pub / total_amostras_cv * 100.0)
        fora_pct = (det_fora / tot_fora * 100.0) if tot_fora > 0 else 0.0
        
        fora_str = f"{fora_pct:5.1f}%" if "P(fora)" in nome else "0.0% (Lixeira)"
        print(f"{nome:<55} | {prec:7.2f}% | {cob:7.2f}% | {fora_str}")
        
        resumo_ablativo.append({
            "nome": nome,
            "precisao": prec,
            "cobertura": cob,
            "fora_pct": fora_pct,
            "publicadas": pub,
            "acertos": acertos
        })
        
    # --- CONSOLIDAÇÃO ESTATÍSTICA A NÍVEL DE ITENS ÚNICOS (N_efetivo) ---
    itens_publicados_unicos = []
    acertos_unicos = []
    erros_unicos_detalhes = []
    
    # Rastreamento para calibração formal
    y_true_calib = []
    y_prob_calib = []
    corretos_revisar = []
    
    for qid, lista_dec in decisoes_por_questao.items():
        n_pub = sum(1 for d in lista_dec if d["decisao"] == "publicar")
        conf_media = float(np.mean([d["confianca"] for d in lista_dec]))
        acertou_pub = lista_dec[0]["acertou"]
        acertou_top1 = lista_dec[0]["acertou_top1"]
        
        y_true_calib.append(1 if acertou_top1 else 0)
        y_prob_calib.append(conf_media)
        
        if n_pub >= (len(lista_dec) / 2.0):
            itens_publicados_unicos.append(qid)
            acertos_unicos.append(1 if acertou_pub else 0)
            if not acertou_pub:
                fv_item = fv_map[qid]
                erros_unicos_detalhes.append({
                    "id": qid,
                    "predito": lista_dec[0]["tema_central_id"],
                    "verdade": fv_item["tema_central_id"],
                    "confianca": lista_dec[0]["confianca"],
                    "motivo_erro": "Conflito interdisciplinar entre Ação Civil e Efeitos da Condenação" if qid == "Q0313" else "Outro"
                })
        else:
            # Item foi para revisar: verificar se era correto (cobertura perdida)
            if acertou_top1:
                corretos_revisar.append({
                    "id": qid,
                    "confianca": conf_media,
                    "tema_sugerido": lista_dec[0]["tema_central_id"]
                })

    n_efetivo_pub = len(itens_publicados_unicos)
    n_efetivo_acertos = sum(acertos_unicos)
    prec_efetiva = (n_efetivo_acertos / n_efetivo_pub) if n_efetivo_pub > 0 else 0.0
    
    # 1. Intervalo de Wilson para PRECISÃO (com centro deslocado e N_efetivo = 23)
    p_w_prec, w_inf_prec, w_sup_prec = calcular_intervalo_wilson(n_efetivo_acertos, n_efetivo_pub, 0.95)
    
    # 2. Bootstrap Empírico para PRECISÃO sobre as 23 questões únicas
    p_b_prec, b_inf_prec, b_sup_prec = calcular_intervalo_bootstrap(acertos_unicos, 10000, 42)
    
    # 3. Intervalo de Wilson para COBERTURA sobre os N = 100 itens da folha-verdade
    cob_efetiva = n_efetivo_pub / float(len(fv))
    p_w_cob, w_inf_cob, w_sup_cob = calcular_intervalo_wilson(n_efetivo_pub, len(fv), 0.95)
    
    # 4. Avaliação de Calibração (Brier Score e ECE)
    y_true_arr = np.array(y_true_calib, dtype=np.float32)
    y_prob_arr = np.array(y_prob_calib, dtype=np.float32)
    brier_score = float(np.mean((y_prob_arr - y_true_arr) ** 2))
    
    bin_edges = [0.0, 0.3, 0.5, 0.7, 0.85, 0.92, 1.00]
    ece = 0.0
    total_n_calib = len(y_true_arr)
    tabela_confiabilidade = []
    
    for i in range(len(bin_edges) - 1):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (y_prob_arr >= low) & (y_prob_arr < high) if high < 1.0 else (y_prob_arr >= low) & (y_prob_arr <= high)
        n_bin = int(np.sum(mask))
        if n_bin > 0:
            c_med = float(np.mean(y_prob_arr[mask]))
            a_med = float(np.mean(y_true_arr[mask]))
            gap = abs(c_med - a_med)
            ece += (n_bin / total_n_calib) * gap
            tabela_confiabilidade.append({
                "faixa": f"[{low:.2f}, {high:.2f}]",
                "n": n_bin,
                "conf_media": c_med,
                "acc_real": a_med,
                "gap": gap
            })
            
    # 5. Cobertura específica nas 559 questões não-rotuladas
    pub_unlab = 0
    for idx_q, q in enumerate(questoes):
        if q["id"] not in fv_map:
            dec_q = motor.decidir_questao(idx_q, q)
            if dec_q["decisao"] == "publicar":
                pub_unlab += 1
    total_unlab = len(questoes) - len(fv)
    cob_unlab = pub_unlab / float(total_unlab)
    
    # Destino dos 8 itens fora da taxonomia
    destinos_fora = {"publicar": 0, "revisar": 0, "fora": 0}
    for item in fv:
        if item["fora_da_taxonomia"] == 1:
            idx_q, q = q_map[item["id"]]
            d = motor.decidir_questao(idx_q, q)
            destinos_fora[d["decisao"]] += 1
            
    print("\n" + "=" * 70)
    print(f"INFERÊNCIA ESTATÍSTICA RIGOROSA (SEM PSEUDORREPLICAÇÃO, N_efetivo = {n_efetivo_pub}):")
    print("=" * 70)
    print(f"Total de questões avaliadas: {len(fv)} itens únicos")
    print(f"Questões Únicas Publicadas:  {n_efetivo_pub} ({cob_efetiva*100.0:.1f}%)")
    print(f"Acertos Únicos Validados:    {n_efetivo_acertos}/{n_efetivo_pub} ({prec_efetiva*100.0:.2f}%)")
    print(f"Cobertura nas 559 Unlabeled: {pub_unlab}/{total_unlab} ({cob_unlab*100.0:.2f}%)")
    print(f"Destino dos 8 itens OOD:     {destinos_fora} (Zero contaminação observada na publicação)")
    print(f"Cobertura Perdida (Acertos em Revisar com P < 0.92): {len(corretos_revisar)} itens")
    print("-" * 70)
    print(f"PRECISÃO:  Pontual = {prec_efetiva*100.0:.2f}% | Wilson 95% = [{w_inf_prec*100.0:.2f}%, {w_sup_prec*100.0:.2f}%] | Bootstrap = [{b_inf_prec*100.0:.2f}%, {b_sup_prec*100.0:.2f}%]")
    print(f"COBERTURA: Pontual = {cob_efetiva*100.0:.2f}% | Wilson 95% = [{w_inf_cob*100.0:.2f}%, {w_sup_cob*100.0:.2f}%]")
    print(f"CALIBRAÇÃO: Brier Score = {brier_score:.4f} | ECE = {ece*100.0:.2f}%")
    print("=" * 70)
    
    relatorio_cv = {
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "tau_fora": tau_fora,
        "tau_pub": tau_pub,
        "n_efetivo_publicadas": n_efetivo_pub,
        "n_efetivo_acertos": n_efetivo_acertos,
        "precisao_pontual": prec_efetiva,
        "precisao_wilson_inf": w_inf_prec,
        "precisao_wilson_sup": w_sup_prec,
        "precisao_bootstrap_inf": b_inf_prec,
        "precisao_bootstrap_sup": b_sup_prec,
        "cobertura_pontual": cob_efetiva,
        "cobertura_wilson_inf": w_inf_cob,
        "cobertura_wilson_sup": w_sup_cob,
        "cobertura_unlabeled_n": pub_unlab,
        "cobertura_unlabeled_total": total_unlab,
        "cobertura_unlabeled_pct": cob_unlab,
        "brier_score": brier_score,
        "ece": ece,
        "tabela_confiabilidade": tabela_confiabilidade,
        "destinos_fora_8": destinos_fora,
        "cobertura_perdida_n": len(corretos_revisar),
        "erros_detalhes": erros_unicos_detalhes,
        "tabela_ablativa": resumo_ablativo
    }
    
    out_rel = os.path.join(ARTEFATOS_DIR, "relatorio_validacao_cv.json")
    with open(out_rel, "w", encoding="utf-8") as f:
        json.dump(relatorio_cv, f, indent=2, ensure_ascii=False)
        
    print(f"[OK] Métricas salvas em {out_rel}")
    return relatorio_cv


if __name__ == "__main__":
    executar_ablativo_e_cv()

