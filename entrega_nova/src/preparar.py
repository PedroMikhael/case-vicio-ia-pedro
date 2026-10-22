"""Preparação e estruturação dos dados e catálogo taxonômico (Case Vício IA)."""

import os
import re
import csv
import json
import html
from typing import Dict, List, Any, Set, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DADOS_DIR = os.path.join(BASE_DIR, "dados")
ARTEFATOS_DIR = os.path.join(BASE_DIR, "artefatos")

os.makedirs(ARTEFATOS_DIR, exist_ok=True)

INACTIVE_THEME_IDS = {"5078", "5234"}

# Dicionário de sinônimos e leis correlacionadas aos temas principais
LEIS_MAP = [
    (r"11\.?343|drogas?|entorpecentes?|tr[aá]fico de drogas", "4708", "Direito Penal - Legislação Especial > Lei de Drogas (Lei 11.343/06)"),
    (r"11\.?340|maria da penha|viol[êe]ncia dom[eé]stica|familiar contra a mulher", "4704", "Direito Penal - Legislação Especial > Lei Maria da Penha (Lei 11.340/06)"),
    (r"8\.?072|crimes? hediondos?", "4690", "Direito Penal - Legislação Especial > Crimes Hediondos (Lei 8.072/90)"),
    (r"7\.?210|execu[cç][ãa]o penal|l\.?e\.?p\.?|remiss[ãa]o de pena|livramento condicional", "4703", "Direito Penal - Legislação Especial > Lei de Execução Penal (Lei 7.210/84)"),
    (r"12\.?850|organiza[cç][õo]es? criminosas?|colabora[cç][ãa]o premiada|infiltra[cç][ãa]o policial", "4705", "Direito Penal - Legislação Especial > Lei das Organizações Criminosas (Lei 12.850/13)"),
    (r"9\.?613|lavagem de dinheiro|branqueamento|oculta[cç][ãa]o de bens", "4702", "Direito Penal - Legislação Especial > Lavagem de Dinheiro (Lei 9.613/98)"),
    (r"10\.?826|estatuto do desarmamento|arma de fogo|porte ilegal|posse irregular", "4694", "Direito Penal - Legislação Especial > Estatuto do Desarmamento (Lei 10.826/03)"),
    (r"13\.?869|abuso de autoridade", "4684", "Direito Penal - Legislação Especial > Nova Lei de Abuso de Autoridade (Lei 13.869/19)"),
    (r"8\.?069|e\.?c\.?a\.?|estatuto da crian[cç]a e do adolescente|ato infracional", "4695", "Direito Penal - Legislação Especial > Estatuto da Criança e do Adolescente (Lei 8.069/90)"),
    (r"9\.?099|juizados? especiais? criminais?|jecrim|transa[cç][ãa]o penal|composi[cç][ãa]o civil|sursis processual", "4701", "Direito Penal - Legislação Especial > Juizados Especiais Criminais (Lei 9.099/95)"),
    (r"9\.?503|c\.?t\.?b\.?|tr[âa]nsito|dire[cç][ãa]o sob a influ[êe]ncia de [aá]lcool|embriaguez ao volante", "4692", "Direito Penal - Legislação Especial > Crimes do Código de Trânsito Brasileiro (Lei 9.503/97)"),
    (r"9\.?605|crimes? ambientais?|meio ambiente", "4688", "Direito Penal - Legislação Especial > Crimes Ambientais (Lei 9.605/98)"),
    (r"10\.?741|estatuto do idoso|estatuto da pessoa idosa", "4696", "Direito Penal - Legislação Especial > Estatuto do Idoso (Lei 10.741/03)"),
    (r"tribunal do j[úu]ri|pron[úu]ncia|impron[úu]ncia|absolvi[cç][ãa]o sum[áa]ria|quesita[cç][ãa]o", "5517", "Processo Penal > Procedimento do Tribunal do Júri"),
    (r"habeas corpus|hc", "5489", "Processo Penal > Habeas Corpus"),
    (r"pris[ãa]o preventiva|pris[ãa]o tempor[áa]ria|pris[ãa]o em flagrante|medidas cautelares", "5524", "Processo Penal > Prisão e Liberdade Provisória"),
    (r"inqu[eé]rito policial|notitia criminis|arquivamento do inqu[eé]rito", "5500", "Processo Penal > Inquérito Policial"),
    (r"compet[êe]ncia territorial|compet[êe]ncia por prerrogativa|conex[ãa]o e contin[êe]ncia", "5480", "Processo Penal > Competência"),
    (r"recurso em sentido estrito|rese|apela[cç][ãa]o criminal|embargos infringentes|agravo em execu[cç][ãa]o", "5560", "Processo Penal > Recursos"),
    (r"prescri[cç][ãa]o|decad[êe]ncia|extin[cç][ãa]o da punibilidade", "5124", "Direito Penal > Extinção da Punibilidade"),
    (r"tentativa|consuma[cç][ãa]o|desist[êe]ncia volunt[áa]ria|arrependimento eficaz|crime imposs[íi]vel", "5112", "Direito Penal > Fato Típico"),
    (r"leg[íi]tima defesa|estado de necessidade|estrito cumprimento do dever|exerc[íi]cio regular", "5151", "Direito Penal > Ilicitude e Causas de Justificação"),
    (r"imputabilidade|potencial consci[êe]ncia|exigibilidade de conduta diversa|erro de proibi[cç][ãa]o", "5079", "Direito Penal > Culpabilidade"),
    (r"dosimetria|crit[eé]rio trif[áa]sico|atenuantes?|agravantes?|causas de aumento", "5199", "Direito Penal > Penas"),
    (r"homic[íi]dio|feminic[íi]dio|infantic[íi]dio|aborto|les[ãa]o corporal", "5100", "Direito Penal > Crimes Contra a Pessoa"),
    (r"furto|roubo|extors[ãa]o|estelionato|apropria[cç][ãa]o ind[eé]bita|recepta[cç][ãa]o", "5157", "Direito Penal > Crimes Contra o Patrimônio"),
    (r"corrup[cç][ãa]o passiva|corrup[cç][ãa]o ativa|peculato|concuss[ãa]o|prevarica[cç][ãa]o", "5087", "Direito Penal > Crimes Praticados Contra a Administração Pública"),
    (r"estupro|ass[eé]dio sexual|importuna[cç][ãa]o sexual", "5110", "Direito Penal > Crimes Contra a Dignidade Sexual"),
    (r"escola cl[áa]ssica|escola positiva|lombroso|ferri|garofalo", "4371", "Criminologia > Etapas Evolutivas e Escolas Criminológicas"),
    (r"labelling approach|etiquetamento|anomia|associa[cç][ãa]o diferencial|subcultura delinquente", "4385", "Criminologia > Teorias Sociológicas Explicativas da Criminalidade"),
    (r"vitimologia|vitimiza[cç][ãa]o prim[áa]ria|vitimiza[cç][ãa]o secund[áa]ria|vitimiza[cç][ãa]o terci[áa]ria", "4366", "Criminologia > Classificação das Vítimas"),
    (r"cifra negra|cifra dourada|cifras criminais", "4379", "Criminologia > Processos de Criminalização e Cifras Criminais > Cifras Criminais"),
    (r"direito penal do inimigo|jakobs", "4376", "Criminologia > Política Criminal > Direito Penal do Inimigo"),
]


def limpar_texto(texto: Optional[str]) -> str:
    """Higieniza HTML cru, decodifica entidades, normaliza quebras e pontuações."""
    if not texto:
        return ""
    t = html.unescape(texto)
    t = t.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\n")
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"<\s*br\s*/?\s*>", "\n", t, flags=re.IGNORECASE)
    t = re.sub(r"<\s*/?(p|div|li|tr|h\d)\s*/?\s*>", "\n", t, flags=re.IGNORECASE)
    t = re.sub(r"<[^>]+>", " ", t)
    
    # Normalizar aspas e travessões
    t = t.replace("“", "\"").replace("”", "\"").replace("’", "'").replace("‘", "'")
    t = t.replace("–", "-").replace("—", "-")
    
    linhas = [re.sub(r"\s+", " ", l).strip() for l in t.split("\n")]
    linhas = [l for l in linhas if l]
    return "\n".join(linhas)


def extrair_sinais_legais(texto: str) -> List[Dict[str, str]]:
    """Detecta citações explícitas de artigos e diplomas legais no texto."""
    matches = []
    texto_lower = texto.lower()
    for padrao, tid, nome in LEIS_MAP:
        if re.search(padrao, texto_lower):
            matches.append({"tema_id": tid, "nome": nome})
    return matches


def carregar_e_processar_questoes() -> List[Dict[str, Any]]:
    """Carrega as 659 questões, higienizando enunciado, comando e alternativas."""
    path_questoes = os.path.join(DADOS_DIR, "questoes.jsonl")
    questoes = []
    
    with open(path_questoes, "r", encoding="utf-8") as f:
        for linha in f:
            linha = linha.strip()
            if not linha:
                continue
            q = json.loads(linha)
            qid = q.get("id")
            
            enunciado = limpar_texto(q.get("enunciado_html", ""))
            comando = limpar_texto(q.get("comando", ""))
            
            # Higienização de alternativas
            alts_raw = q.get("alternativas") or {}
            alts_limpas = {}
            if isinstance(alts_raw, dict):
                for k in sorted(alts_raw.keys()):
                    alts_limpas[k] = limpar_texto(str(alts_raw[k]))
            
            eh_ce = set(alts_limpas.keys()) == {"C", "E"}
            
            # Texto focado: Enunciado + Comando
            if comando and comando not in enunciado:
                texto_busca = f"{enunciado}\n\nComando: {comando}".strip()
            else:
                texto_busca = enunciado.strip()
                
            # Verifica se comando pede incorreta
            pede_incorreta = bool(re.search(r"incorret[ao]|fals[ao]|errad[ao]|n[ãa]o [eé] corret[ao]", texto_busca.lower()))
            
            # Texto completo com alternativas
            partes_comp = [texto_busca]
            if eh_ce:
                partes_comp.append("[Julgamento Certo/Errado]")
            elif alts_limpas:
                partes_comp.append("Alternativas:\n" + "\n".join(f"({k}) {v}" for k, v in alts_limpas.items()))
            texto_completo = "\n\n".join(partes_comp)
            
            sinais = extrair_sinais_legais(texto_busca)
            
            item = {
                "id": qid,
                "texto_busca": texto_busca,
                "texto_completo": texto_completo,
                "enunciado": enunciado,
                "comando": comando,
                "alternativas": alts_limpas,
                "tipo": "certo_errado" if eh_ce else "multipla_escolha",
                "pede_incorreta": pede_incorreta,
                "disciplina_origem": q.get("disciplina_origem") or "",
                "assunto_origem": q.get("assunto_origem") or "",
                "sinais_legais": sinais
            }
            questoes.append(item)
            
    return questoes


def carregar_e_estruturar_arvore() -> Dict[str, Any]:
    """Processa temas.csv, constrói a árvore genealógica e domínios de acerto formal."""
    path_temas = os.path.join(DADOS_DIR, "temas.csv")
    temas = {}
    
    with open(path_temas, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = row["tema_id"]
            temas[tid] = {
                "tema_id": tid,
                "parent_id": row["parent_id"] if row["parent_id"] else None,
                "disciplina": row["disciplina"],
                "nivel": int(row["nivel"]),
                "titulo": row["titulo"],
                "caminho": row["caminho"],
                "descricao": row.get("descricao", ""),
                "status": row.get("status", "active")
            }
            
    # Temas ativos válidos
    temas_ativos = {
        tid: t for tid, t in temas.items() 
        if t["status"] == "active" and tid not in INACTIVE_THEME_IDS
    }
    
    pai_map = {tid: t["parent_id"] for tid, t in temas_ativos.items()}
    filhos_map: Dict[str, List[str]] = {tid: [] for tid in temas_ativos}
    for tid, t in temas_ativos.items():
        pid = t["parent_id"]
        if pid and pid in filhos_map:
            filhos_map[pid].append(tid)
            
    # Disciplina raiz para cada tema ativo
    raiz_map = {}
    for tid, t in temas_ativos.items():
        curr = tid
        while pai_map.get(curr):
            curr = pai_map[curr]
        raiz_map[tid] = curr
        
    # Nós publicáveis (nível 1 e nível 2 ativos, excluindo disciplina nível 0)
    nos_publicaveis = sorted([
        tid for tid, t in temas_ativos.items() if t["nivel"] in (1, 2)
    ])
    
    # Domínio de acerto A(n) para cada nó publicável n:
    # A(n) = {n} U {pais(n) com nivel >= 1} U {filhos diretos de n}
    dominio_acerto: Dict[str, List[str]] = {}
    for n in nos_publicaveis:
        dominio: Set[str] = {n}
        p = pai_map.get(n)
        if p and p in temas_ativos and temas_ativos[p]["nivel"] >= 1:
            dominio.add(p)
        for f in filhos_map.get(n, []):
            dominio.add(f)
        dominio_acerto[n] = sorted(list(dominio))
        
    arvore = {
        "total_temas": len(temas),
        "total_ativos": len(temas_ativos),
        "total_publicaveis": len(nos_publicaveis),
        "inativos_proibidos": list(INACTIVE_THEME_IDS),
        "temas": temas_ativos,
        "pai_map": pai_map,
        "filhos_map": filhos_map,
        "raiz_map": raiz_map,
        "nos_publicaveis": nos_publicaveis,
        "dominio_acerto": dominio_acerto
    }
    
    out_catalogo = os.path.join(ARTEFATOS_DIR, "catalogo_arvore.json")
    with open(out_catalogo, "w", encoding="utf-8") as f:
        json.dump(arvore, f, ensure_ascii=False, indent=2)
        
    print(f"[OK] Catálogo da árvore gerado em {out_catalogo}")
    print(f"     - Total ativos: {len(temas_ativos)}")
    print(f"     - Total publicáveis (nível 1 e 2): {len(nos_publicaveis)}")
    
    return arvore


def carregar_folha_verdade() -> List[Dict[str, Any]]:
    """Carrega o ground-truth de 100 questões."""
    path_fv = os.path.join(DADOS_DIR, "folha_verdade.csv")
    registros = []
    with open(path_fv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            registros.append({
                "id": r["id"],
                "tema_central_id": r["tema_central_id"],
                "tema_central_caminho": r.get("tema_central_caminho", ""),
                "extras_ids": [x.strip() for x in r["extras_ids"].split(";") if x.strip()] if r.get("extras_ids") else [],
                "fora_da_taxonomia": int(r["fora_da_taxonomia"]) if r.get("fora_da_taxonomia") else 0
            })
    return registros


if __name__ == "__main__":
    print("=== [PREPARAR] Processando dados e estruturando a árvore taxonômica ===")
    arvore = carregar_e_estruturar_arvore()
    questoes = carregar_e_processar_questoes()
    fv = carregar_folha_verdade()
    print(f"[OK] {len(questoes)} questões carregadas e higienizadas.")
    print(f"[OK] {len(fv)} registros da folha-verdade carregados.")
