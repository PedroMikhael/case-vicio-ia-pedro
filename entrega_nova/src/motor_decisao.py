"""Motor de Decisão Bayesiana na Árvore e Calibração de Probabilidades (Case Vício IA)."""

import os
import json
import numpy as np
from typing import Dict, List, Any, Tuple, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARTEFATOS_DIR = os.path.join(BASE_DIR, "artefatos")

DISCIPLINAS_ESTRANHAS = {
    "direito tributário", "direito tributario", "tributário", "tributario",
    "direito civil", "civil", "direito empresarial", "empresarial",
    "direito administrativo", "administrativo", "direito previdenciário", "previdenciário",
    "direito constitucional", "direito do trabalho", "trabalho", "processo civil",
    "direito militar", "penal militar", "processo penal militar"
}

# Disciplinas válidas mapeadas para os nós raiz
RAIZ_DISCIPLINA_MAP = {
    "penal": "5062",
    "direito penal": "5062",
    "processo penal": "5465",
    "direito processual penal": "5465",
    "direito penal - legislação especial": "4683",
    "legislação penal especial": "4683",
    "legislação especial": "4683",
    "leis penais especiais": "4683",
    "criminologia": "4365"
}


class MotorDecisao:
    """Motor de decisão e inferência probabilística com árvore genealógica."""

    def __init__(
        self,
        catalogo_arvore_path: Optional[str] = None,
        embeddings_cache_path: Optional[str] = None,
        temperatura: float = 0.016,
        tau_fora: float = 0.40,
        tau_pub: float = 0.92,
        w_dense: float = 0.55,
        w_sparse: float = 0.45,
        w_symb: float = 0.0,
        w_origem: float = 0.0
    ):
        self.temperatura = temperatura
        self.tau_fora = tau_fora
        self.tau_pub = tau_pub
        self.w_dense = w_dense
        self.w_sparse = w_sparse
        self.w_symb = w_symb
        self.w_origem = w_origem
        
        arv_path = catalogo_arvore_path or os.path.join(ARTEFATOS_DIR, "catalogo_arvore.json")
        emb_path = embeddings_cache_path or os.path.join(ARTEFATOS_DIR, "embeddings_cache.npz")
        
        with open(arv_path, "r", encoding="utf-8") as f:
            self.arvore = json.load(f)
            
        self.temas = self.arvore["temas"]
        self.pai_map = self.arvore["pai_map"]
        self.filhos_map = self.arvore["filhos_map"]
        self.raiz_map = self.arvore["raiz_map"]
        self.dominio_acerto = self.arvore["dominio_acerto"]
        
        # Carrega embeddings pré-calculados
        cache = np.load(emb_path)
        self.questoes_ids = [str(x) for x in cache["questoes_ids"]]
        self.questoes_dense = cache["questoes_dense"]
        self.temas_ids = [str(x) for x in cache["temas_ids"]]
        self.temas_dense = cache["temas_dense"]
        self.inativos_ids = [str(x) for x in cache["inativos_ids"]]
        self.inativos_dense = cache["inativos_dense"]
        self.sparse_sim = cache["sparse_sim"]
        self.inativos_sparse_sim = cache["inativos_sparse_sim"]
        
        # Mapeamento estritamente canônico pelas colunas de temas_ids
        self.nos_publicaveis = list(self.temas_ids)
        self.tema_idx_map = {tid: i for i, tid in enumerate(self.temas_ids)}
        self.idx_to_tema = {i: tid for i, tid in enumerate(self.temas_ids)}
        self.n_temas = len(self.temas_ids)
        
        # Pré-computa produto escalar denso: (659, 371)
        self.dense_sim = self.questoes_dense @ self.temas_dense.T
        
        # Similaridade com inativos: (659, 2)
        if len(self.inativos_dense) > 0:
            self.inativos_dense_sim = self.questoes_dense @ self.inativos_dense.T
        else:
            self.inativos_dense_sim = np.zeros((len(self.questoes_ids), 0), dtype=np.float32)
            
        # Matriz binária de domínios de acerto: A_mat[n, t] = 1 se t in A(n)
        self.A_mat = np.zeros((self.n_temas, self.n_temas), dtype=np.float32)
        for i, n in enumerate(self.temas_ids):
            for t in self.dominio_acerto.get(n, [n]):
                if t in self.tema_idx_map:
                    self.A_mat[i, self.tema_idx_map[t]] = 1.0

    def calcular_scores_questao(self, q_idx: int, questao: Dict[str, Any]) -> np.ndarray:
        """Combina os canais denso, esparso, simbólico e de origem para os 371 temas."""
        # 1. Canal Denso
        s_dense = self.dense_sim[q_idx].copy()
        
        # 2. Canal Esparso (TF-IDF/BM25)
        s_sparse = self.sparse_sim[q_idx].copy()
        
        # 3. Canal Simbólico (Match determinístico de leis e artigos)
        s_symb = np.zeros(self.n_temas, dtype=np.float32)
        sinais = questao.get("sinais_legais", [])
        for sinal in sinais:
            tid_alvo = sinal["tema_id"]
            if tid_alvo in self.tema_idx_map:
                idx = self.tema_idx_map[tid_alvo]
                s_symb[idx] += 1.0
                # Bônus para filhos do tema de lei
                for filho in self.filhos_map.get(tid_alvo, []):
                    if filho in self.tema_idx_map:
                        s_symb[self.tema_idx_map[filho]] += 0.8
                        
        # 4. Canal de Origem (Compatibilidade com a disciplina raiz)
        s_orig = np.zeros(self.n_temas, dtype=np.float32)
        disc_orig = (questao.get("disciplina_origem") or "").lower().strip()
        raiz_esperada = RAIZ_DISCIPLINA_MAP.get(disc_orig)
        if raiz_esperada:
            for i, tid in enumerate(self.nos_publicaveis):
                if self.raiz_map.get(tid) == raiz_esperada:
                    s_orig[i] = 1.0
                else:
                    s_orig[i] = -0.5
                    
        # Combinação ponderada
        scores = (
            self.w_dense * s_dense +
            self.w_sparse * s_sparse +
            self.w_symb * s_symb +
            self.w_origem * s_orig
        )
        return scores

    def calibrar_probabilidades(self, scores: np.ndarray, temperatura: Optional[float] = None) -> np.ndarray:
        """Aplica Softmax com temperatura calibrada para obter p(y = t | x)."""
        t = temperatura or self.temperatura
        shift_scores = (scores - np.max(scores)) / max(0.001, t)
        exp_scores = np.exp(np.clip(shift_scores, -50.0, 50.0))
        return exp_scores / np.sum(exp_scores)

    def maximizar_probabilidade_arvore(self, probs: np.ndarray) -> Tuple[str, float, int]:
        """Calcula P(acerto | n) = sum_{t in A(n)} p(y = t) e seleciona o nó ótimo n*."""
        # Multiplicação matricial: p_acerto = A_mat @ probs
        p_acerto = self.A_mat @ probs
        melhor_idx = int(np.argmax(p_acerto))
        n_estrela = self.idx_to_tema[melhor_idx]
        p_estrela = float(p_acerto[melhor_idx])
        return n_estrela, p_estrela, melhor_idx

    def estimar_probabilidade_fora(
        self,
        q_idx: int,
        questao: Dict[str, Any],
        scores: np.ndarray,
        p_estrela: float
    ) -> Tuple[float, str]:
        """Modela explicitamente a probabilidade de a questão ser Out-of-Distribution (OOD)."""
        score_max = float(np.max(scores))
        score_dense_max = float(np.max(self.dense_sim[q_idx]))
        
        # 1. Proximidade com temas inativos (ex: 5078 - Direito Penal Econômico, 5234 - Tratados)
        inat_dense_max = float(np.max(self.inativos_dense_sim[q_idx])) if self.inativos_dense_sim.shape[1] > 0 else 0.0
        inat_sparse_max = float(np.max(self.inativos_sparse_sim[q_idx])) if self.inativos_sparse_sim.shape[1] > 0 else 0.0
        score_inativo = 0.7 * inat_dense_max + 0.3 * inat_sparse_max
        
        # 2. Disciplina de origem manifestamente fora do escopo penal
        disc_orig = (questao.get("disciplina_origem") or "").lower().strip()
        assunto_orig = (questao.get("assunto_origem") or "").lower().strip()
        
        eh_disc_estranha = any(d in disc_orig or d in assunto_orig for d in DISCIPLINAS_ESTRANHAS)
        sinais_penais = len(questao.get("sinais_legais", [])) > 0
        
        # 1. Filtro Curricular Estrito: Ramo Militar (CPM/CPPM)
        # O catálogo da plataforma cobre exclusivamente Direito Penal Comum, Processual Penal Comum e Leis Penais Especiais.
        # Crimes militares e processo penal militar (Decretos-Leis 1.001/69 e 1.002/69) constituem ramo especializado alheio.
        eh_militar = any(m in disc_orig or m in assunto_orig for m in ["militar", "cpm", "cppm", "ipm"])
        if eh_militar:
            return 0.95, "Matéria de Direito Penal Militar / Processual Penal Militar (CPM/CPPM) fora da taxonomia penal comum da plataforma"

        # 2. Foco em tema inativo proibido (ex: 5078 - Direito Penal Econômico, 5234 - Tratados)
        if score_inativo > (score_dense_max - 0.02) and score_inativo > 0.45:
            return 0.85, "Matéria aderente a tema inativo da taxonomia (ex: Direito Penal Econômico/Tratados)"
            
        # 3. Disciplina de origem manifestamente alheia sem respaldo penal
        if any(d in disc_orig for d in ["direito tributário", "tributario"]) and not sinais_penais:
            return 0.35, f"Origem fora do escopo penal/criminológico ({disc_orig}) sem conexão temática"
            
        # 4. Baixa densidade semântica geral (outlier distante de toda a taxonomia)
        if score_dense_max < 0.28:
            return 0.65, "Baixa aderência conceitual com qualquer tema ativo do catálogo penal"
            
        # 5. Ambiguidade difusa e acerto esperado fraco
        if p_estrela < 0.35 and score_dense_max < 0.36:
            return 0.52, "Incerteza semântica extrema com hipótese plausível de matéria fora do catálogo"
            
        return 0.05, ""

    def extrair_temas_extras(
        self,
        n_estrela: str,
        probs: np.ndarray,
        limiar_prob: float = 0.12
    ) -> str:
        """Identifica temas secundários legítimos fora do ramo principal."""
        ramo_n = set(self.dominio_acerto.get(n_estrela, [n_estrela]))
        extras = []
        for i, p in enumerate(probs):
            if p >= limiar_prob:
                tid = self.idx_to_tema[i]
                if tid not in ramo_n and tid in self.temas:
                    extras.append(tid)
        return ";".join(sorted(extras))

    def decidir_questao(self, q_idx: int, questao: Dict[str, Any]) -> Dict[str, Any]:
        """Executa a política de decisão ótima em três estágios para uma questão."""
        qid = questao["id"]
        
        # 1. Scores e Probabilidades Posteriores
        scores = self.calcular_scores_questao(q_idx, questao)
        probs = self.calibrar_probabilidades(scores)
        
        # 2. Maximização de P(acerto) na Árvore
        n_estrela, p_estrela, melhor_idx = self.maximizar_probabilidade_arvore(probs)
        
        # 3. Modelagem de P(fora)
        p_fora, motivo_fora = self.estimar_probabilidade_fora(q_idx, questao, scores, p_estrela)
        
        # 4. Portão B de Concordância de Raiz/Disciplina entre Denso e Esparso
        top_dense = self.idx_to_tema[int(np.argmax(self.dense_sim[q_idx]))]
        top_sparse = self.idx_to_tema[int(np.argmax(self.sparse_sim[q_idx]))]
        raiz_dense = self.temas.get(top_dense, {}).get("disciplina")
        raiz_sparse = self.temas.get(top_sparse, {}).get("disciplina")
        concordancia_raiz = (raiz_dense == raiz_sparse)
        
        # 5. Temas extras abordados
        extras_ids = self.extrair_temas_extras(n_estrela, probs)
        
        # Regra de Decisão Oficial com dois limiares e Portão B de Concordância:
        # A) Se P(fora) >= tau_fora -> 'fora'
        if p_fora >= self.tau_fora:
            return {
                "id": qid,
                "decisao": "fora",
                "tema_central_id": "",
                "extras_ids": "",
                "confianca": round(float(p_fora), 2),
                "motivo": motivo_fora[:150]
            }
            
        # B) Senão, se P*(acerto) >= tau_pub E concordância de disciplina raiz -> 'publicar' em n*
        elif p_estrela >= self.tau_pub and concordancia_raiz:
            return {
                "id": qid,
                "decisao": "publicar",
                "tema_central_id": n_estrela,
                "extras_ids": extras_ids,
                "confianca": round(float(p_estrela), 2),
                "motivo": ""
            }
            
        # C) Senão -> 'revisar' com sugestão n*
        else:
            tema_nome = self.temas[n_estrela]["titulo"]
            if p_estrela >= self.tau_pub and not concordancia_raiz:
                motivo_rev = f"Concorrência interdisciplinar entre canal denso ({raiz_dense}) e esparso ({raiz_sparse}) sob P*={p_estrela:.2f}"
            else:
                # Identifica segundo colocado
                scores_sem_max = scores.copy()
                scores_sem_max[melhor_idx] = -999.0
                seg_idx = int(np.argmax(scores_sem_max))
                seg_nome = self.temas[self.idx_to_tema[seg_idx]]["titulo"]
                motivo_rev = f"Dúvida limítrofe entre '{tema_nome}' e '{seg_nome}' (P*={p_estrela:.2f})"
                
            return {
                "id": qid,
                "decisao": "revisar",
                "tema_central_id": n_estrela,
                "extras_ids": extras_ids,
                "confianca": round(float(p_estrela), 2),
                "motivo": motivo_rev[:150]
            }
