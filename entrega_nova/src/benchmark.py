"""Medição de Latência e Throughput de Inferência (Case Vício IA).

Mede o throughput da Etapa 2 (Decisão matricial e propagação na árvore)
em ambiente local single-thread, salvando os resultados em JSON.
"""

import os
import sys

# Força execução single-thread no BLAS antes de carregar numpy
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import json
import platform
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.motor_decisao import MotorDecisao
from src.preparar import carregar_e_processar_questoes


def coletar_informacoes_hardware():
    return {
        "so": platform.platform(),
        "processador": platform.processor() or "x86_64",
        "cpus_logicas": os.cpu_count() or 1,
        "threads_blas": int(os.environ.get("OMP_NUM_THREADS", "1")),
        "versao_python": sys.version.split()[0],
        "versao_numpy": np.__version__
    }


def executar_benchmark(n_repeticoes: int = 10, output_json: str = None):
    print("=" * 68)
    print("BENCHMARK OFICIAL DE INFERÊNCIA E LATÊNCIA (ETAPA 2: DECISÃO)")
    print("=" * 68)
    
    hw = coletar_informacoes_hardware()
    print("Ambiente de Execução Local:")
    print(f"  - Sistema Operacional: {hw['so']}")
    print(f"  - Processador:         {hw['processador']}")
    print(f"  - CPUs Lógicas:        {hw['cpus_logicas']} (Execução limitada a {hw['threads_blas']} thread BLAS)")
    print(f"  - Python / NumPy:      Python {hw['versao_python']} | NumPy {hw['versao_numpy']}")
    print("-" * 68)

    # 1. Carregamento dos dados e inicialização
    t0 = time.perf_counter()
    questoes = carregar_e_processar_questoes()
    total_q = len(questoes)
    t_carga_dados = time.perf_counter() - t0

    t0 = time.perf_counter()
    motor = MotorDecisao()
    t_carga_artefatos = time.perf_counter() - t0

    print(f"1. Carga de {total_q} questões higienizadas: {t_carga_dados*1000:.1f} ms")
    print(f"2. Carga dos artefatos em memória (NPZ + Árvore JSON): {t_carga_artefatos*1000:.1f} ms")

    # 2. Warm-up
    for q_idx, q in enumerate(questoes[:50]):
        _ = motor.decidir_questao(q_idx, q)

    # 3. Medição da Etapa 2 (Decisão em Memória)
    tempos_etapa2 = []
    print(f"\n3. Executando {n_repeticoes} passadas completas da Etapa 2 sobre as {total_q} questões...")
    for _ in range(n_repeticoes):
        t_inicio = time.perf_counter()
        for q_idx, q in enumerate(questoes):
            _ = motor.decidir_questao(q_idx, q)
        t_fim = time.perf_counter()
        tempos_etapa2.append(t_fim - t_inicio)

    tempos_etapa2 = np.array(tempos_etapa2)
    tempo_etapa2_medio = float(np.mean(tempos_etapa2))
    tempo_etapa2_std = float(np.std(tempos_etapa2))
    throughput_etapa2 = total_q / tempo_etapa2_medio
    latencia_etapa2_ms = (tempo_etapa2_medio / total_q) * 1000.0

    # 4. Projeção para 50.000 questões
    tempo_50k_etapa2_s = 50000.0 / throughput_etapa2

    print("-" * 68)
    print("Resultados da Etapa 2 (Decisão Matricial + Árvore + Portão B):")
    print(f"  - Tempo Médio por Passada: {tempo_etapa2_medio*1000:.1f} ms (± {tempo_etapa2_std*1000:.1f} ms)")
    print(f"  - Throughput Medido:       {throughput_etapa2:.1f} questões/segundo")
    print(f"  - Latência Unitária:       {latencia_etapa2_ms:.3f} ms/questão")
    print(f"  - Projeção para 50.000 q:  {tempo_50k_etapa2_s:.1f} segundos (~{tempo_50k_etapa2_s/60:.2f} min)")
    print("  - Nota Metodológica:       Mede a inferência probabilística e árvore em memória;")
    print("                             não inclui transform TF-IDF / embedding de textos novos.")
    print("=" * 68)

    resultado = {
        "hardware": hw,
        "total_questoes_amostra": total_q,
        "n_repeticoes": n_repeticoes,
        "etapa2_decisao": {
            "tempo_medio_passada_ms": round(tempo_etapa2_medio * 1000.0, 2),
            "desvio_padrao_ms": round(tempo_etapa2_std * 1000.0, 2),
            "throughput_questoes_por_segundo": round(throughput_etapa2, 1),
            "latencia_por_questao_ms": round(latencia_etapa2_ms, 3),
            "projecao_tempo_50k_segundos": round(tempo_50k_etapa2_s, 1),
            "projecao_tempo_50k_minutos": round(tempo_50k_etapa2_s / 60.0, 2)
        },
        "etapa1_estimativas_documentadas": {
            "gpu_t4_fp16_batch_q_por_s": 1100,
            "gpu_t4_tempo_50k_segundos": 45.0,
            "cpu_pura_q_por_s": 35,
            "cpu_pura_tempo_50k_minutos": 24.0
        }
    }

    out_path = output_json or os.path.join(BASE_DIR, "artefatos", "benchmark_resultado.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)
    print(f"[OK] Resultados salvos em {out_path}")

    return resultado


if __name__ == "__main__":
    executar_benchmark(n_repeticoes=10)
