"""Geração offline de artefatos: catálogo lexical enriquecido e embeddings vetoriais densos/esparsos via intfloat/multilingual-e5-base."""

import os
import sys
import csv
import json
import re
import numpy as np
from typing import Dict, List, Any, Tuple

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)

from src.preparar import carregar_e_processar_questoes, carregar_e_estruturar_arvore

DADOS_DIR = os.path.join(BASE_DIR, "dados")
ARTEFATOS_DIR = os.path.join(BASE_DIR, "artefatos")

# Dicionário de conhecimento jurídico especializado e cirúrgico por tema
CONHECIMENTO_JURIDICO = {
    # Penal Geral
    "5064": "lei penal no tempo, principio da legalidade, anterioridade, retroatividade benefica, abolitio criminis, novatio legis in pejus, territorialidade, extraterritorialidade, lugar do crime, tempo do crime, art. 1 a 12 CP",
    "5112": "fato tipico, conduta, dolo, culpa, preterdolo, nexo causal, tipicidade formal e material, principio da insignificancia, iter criminis, tentativa, desistencia voluntaria, arrependimento eficaz, arrependimento posterior, crime impossivel, erro de tipo, art. 13 a 22 CP",
    "5151": "ilicitude, antijuridicidade, excludentes de ilicitude, legitima defesa, estado de necessidade, estrito cumprimento do dever legal, exercicio regular de direito, excesso punivel, art. 23 a 25 CP",
    "5079": "culpabilidade, imputabilidade penal, menoridade penal, doenca mental, embriaguez, potencial consciencia da ilicitude, erro de proibicao, exigibilidade de conduta diversa, coacao moral irresistivel, obediencia hierarquica, art. 21 a 28 CP",
    "5082": "concurso de pessoas, concurso de agentes, coautoria, participacao, instigacao, auxilio, autoria mediata, cooperacao dolosamente distinta, art. 29 a 31 CP",
    "5081": "concurso de crimes, concurso material, concurso formal, crime continuado, continuidade delitiva, cumulo material, exasperacao, art. 69 a 71 CP",
    "5199": "penas, dosimetria da pena, criterio trifasico, circunstancias judiciais art. 59, atenuantes e agravantes arts. 61 a 66, reincidencia, causas de aumento e diminuicao, penas restritivas de direitos, multa, sursis, livramento condicional",
    "5155": "medidas de seguranca, internacao, tratamento ambulatorial, desinternacao condicional, semi-imputabilidade, art. 96 a 99 CP",
    "5124": "extinção da punibilidade, prescricao da pretensao punitiva, prescricao da pretensao executoria, prescricao retroativa, decadencia, perempcao, morte do agente, anistia, graca, indulto, perdao judicial, renuncia, retratacao, art. 107 a 120 CP",
    "5063": "acao penal penal substantiva, acao publica incondicionada, condicionada a representacao, acao penal privada, decadencia, renuncia, perdao, art. 100 a 106 CP",
    "5175": "iter criminis, fases do crime, cogitacao, atos preparatorios impuniveis, atos executivos, consumacao, tentativa perfeita e imperfeita conatus",
    "5180": "lei penal no tempo e no espaco, principio da legalidade, ultratividade, retroatividade, lugar do crime",
    "5183": "conflito aparente de normas penais, principio da especialidade, consuncao crime meio e crime fim, subsidiariedade, alternatividade",
    "5184": "medida de seguranca, inimputaveis e semi-imputaveis, internacao em hospital de custodia, desinternacao condicional",
    
    # Penal Especial
    "5100": "crimes contra a pessoa, homicidio, feminicidio, infanticidio, induzimento ao suicidio, aborto, lesao corporal, periclitação da vida, calunia, difamacao, injuria, ameaca, sequestro, art. 121 a 154 CP",
    "5150": "art. 121 CP, homicidio simples, homicidio qualificado, feminicidio, motivo futil, torpe, meio cruel, asfixia, emboscada, recurso que impossibilitou a defesa",
    "5149": "art. 123 CP, infanticidio, sob influencia do estado puerperal matar o proprio filho durante o parto ou logo apos",
    "5152": "art. 122 CP, induzimento instigacao ou auxilio a suicidio ou automutilacao",
    "5148": "art. 124 a 128 CP, aborto provocado pela gestante, aborto de terceiro com ou sem consentimento, aborto legal terapeutico sentimental",
    "5145": "art. 129 CP, lesao corporal leve, grave, gravissima, seguida de morte, violencia domestica",
    "5157": "crimes contra o patrimonio, furto, roubo, extorsao, apropriacao indebita, estelionato, receptacao, dano, art. 155 a 183 CP",
    "5159": "art. 155 CP, furto simples, furto qualificado, repouso noturno, rompimento de obstaculo, chave falsa, furto mediante fraude",
    "5160": "art. 157 CP, roubo simples, roubo majorado arma de fogo, latrocinio roubo qualificado pela morte, roubo proprio e improprio",
    "5166": "art. 171 CP, estelionato, vantagem ilicita em prejuizo alheio, induzindo ou mantendo em erro mediante artificio ardil, fraude eletronica",
    "5161": "art. 158 CP, extorsao, extorsao mediante sequestro art. 159, sequestro relampago",
    "5165": "art. 168 CP, apropriacao indebita, apropriacao indebita previdenciaria art. 168-A",
    "5169": "art. 180 CP, receptacao simples, receptacao qualificada, receptacao culposa, produto de crime",
    "5164": "art. 163 CP, dano simples, dano qualificado, destruicao de patrimonio publico",
    "5110": "crimes contra a dignidade sexual, estupro art. 213, estupro de vulneravel art. 217-A, importunacao sexual art. 215-A, assedio sexual art. 216-A",
    "5099": "crimes contra a familia, bigamia, abandono material, abandono intelectual, art. 235 a 249 CP",
    "5104": "crimes contra a incolumidade publica, incendio, explosao, crimes contra a saude publica, epidemia, falsificacao de medicamentos",
    "5107": "crimes contra a paz publica, incitacao ao crime, apologia, associacao criminosa art. 288, milicia privada",
    "5101": "crimes contra a fe publica, moeda falsa, falsidade documental, documento publico particular, falsidade ideologica art. 299, uso de documento falso art. 304",
    "5117": "dos crimes contra a fe publica, moeda falsa, falsificacao de documentos, falsidade ideologica, certidao falsa, uso de documento falso",
    "5087": "crimes contra a administracao publica, peculato, concussao, corrupcao passiva, prevaricacao, desacato, desobediencia, corrupcao ativa, denunciação caluniosa, falso testemunho",
    "5090": "art. 312 CP, peculato apropriacao, peculato desvio, peculato furto, peculato culposo reparacao do dano",
    "5091": "art. 316 CP, concussao, exigir vantagem indevida em razao da funcao publica",
    "5092": "art. 317 CP, corrupcao passiva, solicitar ou receber vantagem indevida ou aceitar promessa",
    "5097": "art. 333 CP, corrupcao ativa, oferecer ou prometer vantagem indevida a funcionario publico",
    "5095": "art. 319 CP, prevaricacao, retardar ou deixar de praticar ato de oficio para satisfazer sentimento pessoal",
    "5096": "art. 331 CP, desacato a funcionario publico no exercicio da funcao",
    "5098": "art. 339 CP, denunciacao caluniosa, imputar crime a quem sabe inocente dando causa a investigacao",
    "5094": "art. 342 CP, falso testemunho, falsa pericia, afirmacao falsa de testemunha perito tradutor",
    
    # Processo Penal
    "5500": "inquerito policial, policia judiciaria, delegado, notitia criminis, instauracao, portaria, arquivamento art. 28 CPP, indiciamento, prazos, valor probatorio",
    "5466": "acao penal processual, acao publica incondicionada, condicionada a representacao, acao penal privada, denuncia, queixa-crime, rejeicao liminar art. 395 CPP",
    "5480": "competencia processual penal, lugar da infracao, domicilio do reu, distribuicao, prevencao, conexao, continencia, foro por prerrogativa de funcao",
    "5535": "provas processo penal, teoria da prova, exame de corpo de delito, interrogatorio, confissao, testemunhas, contradita, busca e apreensao, prova ilicita",
    "5543": "provas em especie, corpo de delito, interrogatorio do acusado, confissao, testemunhas, documentos, pericia, reconhecimento de pessoas art. 226 CPP",
    "5524": "procedimento do tribunal do juri, crimes dolosos contra a vida, rito bifasico, sumario de culpa e plenario, jurados, quesitos, soberania dos veredictos",
    "5527": "art. 413 a 419 CPP, pronuncia indicios de autoria, impronuncia ausencia de prova, absolvicao sumaria excludente, desclassificacao",
    "5519": "procedimento comum, rito ordinario, rito sumario, rito sumarissimo, resposta a acusacao, audiencia de instrucao",
    "5560": "recursos em processo penal, recursos em especies, apelacao criminal, recurso em sentido estrito RESE, agravo em execucao, embargos infringentes, embargos de declaracao, carta testemunhavel",
    "5562": "art. 593 CPP, recurso de apelacao criminal, sentenca definitiva absolutoria ou condenatoria, apelacao no tribunal do juri, absolvicao sumaria",
    "5561": "art. 581 CPP, recurso em sentido estrito, RESE, decisao que rejeitar denuncia, pronunciar o reu, conceder ou negar fianca, juizo de retratacao",
    "5566": "art. 197 LEP, agravo em execucao penal, recurso contra decisoes do juiz da execucao penal",
    "5477": "art. 647 a 667 CPP, habeas corpus, coacao ilegal, liberdade de locomocao, prisao ilegal, excesso de prazo, falta de justa causa, liminar",
    "5489": "habeas corpus garantia constitucional e processual penal, coacao ilegal a liberdade de locomocao, prisao arbitraria, excesso de prazo",
    "5523": "prisao e liberdade provisoria, prisao em flagrante, audiencia de custodia, prisao preventiva art. 312 CPP, garantia da ordem publica, prisao temporaria, medidas cautelares art. 319",
    "5487": "comunicacao dos atos processuais, citacao pessoal, citacao por edital, citacao hora certa, intimacoes, notificacoes no processo penal",
    
    # Legislação Especial
    "4708": "Lei 11.343/06, Lei de Drogas, trafico de drogas art. 33 caput, trafico privilegiado parágrafo 4, posse e uso de drogas art. 28, associacao para o trafico art. 35",
    "4704": "Lei 11.340/06, Lei Maria da Penha, violencia domestica e familiar contra a mulher, medidas protetivas de urgencia, descumprimento art. 24-A",
    "4690": "Lei 8.072/90, Crimes Hediondos, rol taxativo, tortura, trafico, terrorismo, homicidio qualificado, latrocinio, progressao de regime, inafiancavel",
    "4703": "Lei 7.210/84, Lei de Execucao Penal, LEP, remicao de pena por trabalho e estudo, progressao de regime, saida temporaria, faltas disciplinares, RDD",
    "4705": "Lei 12.850/13, Organizacoes Criminosas, conceito de organizacao criminosa, 4 ou mais pessoas, colaboracao premiada, delacao, acao controlada, infiltracao policial",
    "4702": "Lei 9.613/98, Lavagem de Dinheiro, colocacao, dissimulacao, integracao, ocultacao de bens valores, infracao antecedente, crime autonomo",
    "4694": "Lei 10.826/03, Estatuto do Desarmamento, posse irregular de arma art. 12, porte ilegal de arma art. 14, arma de uso restrito ou proibido art. 16",
    "4684": "Lei 13.869/19, Abuso de Autoridade, agente publico servidor, dolo especifico de prejudicar outrem ou beneficiar a si por capricho, vedado crime de hermeneutica",
    "4701": "Lei 9.099/95, Juizados Especiais Criminais, JECRIM, infracao de menor potencial ofensivo ate 2 anos, TCO, transacao penal, composicao civil, sursis processual art. 89",
    "4695": "Lei 8.069/90, Estatuto da Crianca e do Adolescente, ECA, ato infracional, medidas socioeducativas, internacao, crimes contra crianca e adolescente, pornografia infantil",
    "4692": "Lei 9.503/97, Codigo de Transito Brasileiro, CTB, embriaguez ao volante art. 306, homicidio culposo na direcao art. 302, racha disputa automobilistica",
    "4688": "Lei 9.605/98, Crimes Ambientais, tutela penal do meio ambiente, crimes contra fauna, flora, poluicao, responsabilidade penal de pessoa juridica",
    "4696": "Lei 10.741/03, Estatuto do Idoso, crimes contra a pessoa idosa, abandono, apropriacao de bens de idoso",
    "4706": "Lei 13.260/16, Terrorismo, atos de terrorismo, motivacao de xenofobia discriminacao preconceito de raca cor etnia religiao",
    
    # Criminologia
    "4366": "vitimologia, classificacao das vitimas de Benjamin Mendelsohn, vitima ideal inocente, vitima culpada, vitima provocadora, vitimizacao primaria secundaria terciaria",
    "4367": "classificacao dos criminosos, Cesare Lombroso, criminoso nato atavismo, criminoso louco, criminoso de ocasiao, Enrico Ferri, Raffaele Garofalo",
    "4371": "escolas criminologicas, escola classica Cesare Beccaria livre arbitrio proporcionalidade, escola positiva Lombroso Ferri Garofalo determinismo empirico, terceira escola",
    "4374": "politica criminal, direito penal do inimigo Gunther Jakobs, garantismo penal Luigi Ferrajoli, abolicionismo penal, prevencao geral e especial",
    "4377": "cifras criminais, cifra negra criminalidade oculta, cifra dourada crimes de colarinho branco, criminalizacao primaria e secundaria",
    "4380": "teoria geral da criminologia, ciencia empirica indutiva, objeto da criminologia delito delinquente vitima controle social",
    "4385": "teorias sociologicas criminologia, teoria da anomia Merton Durkheim, associacao diferencial Sutherland crimes de colarinho branco, labelling approach etiquetamento Becker Lemert, subcultura delinquente Cohen, escola de chicago",
    "4368": "criminologia no estado democratico de direito, prevencao do delito, reacao social, controle da criminalidade",
    "4389": "vitimologia, estudo da vitima no crime, vitimizacao primaria secundaria terciaria, iter victimae",
    
    # Inativos
    "5078": "direito penal economico, crimes contra a ordem economica, crimes contra o sistema financeiro nacional lei 7492, crimes tributarios sonegacao fiscal lei 8137, cartel",
    "5234": "tratados e convencoes em materia criminal internacional, cooperacao juridica internacional, extradicao, tribunal penal internacional TPI, cartas rogatorias"
}


def enriquecer_temas(arvore: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Gera dados enriquecidos específicos para cada tema."""
    temas = arvore["temas"]
    linhas_enriquecidas = []
    
    for tid, t in temas.items():
        nivel = t["nivel"]
        tit = t["titulo"]
        caminho = t["caminho"]
        disc = t["disciplina"]
        desc = t.get("descricao", "")
        
        desc_limpa = re.sub(r"\d+\s*questões no source\.\s*Path:[^.]*\.?", "", desc, flags=re.IGNORECASE).strip()
        desc_limpa = re.sub(r"Subtema de '[^']*'\s*\([^)]*\)\.?", "", desc_limpa, flags=re.IGNORECASE).strip()
        desc_limpa = desc_limpa.strip()
        
        conhecimento = CONHECIMENTO_JURIDICO.get(tid, "")
        
        partes = [
            f"Disciplina: {disc}",
            f"Ramo: {caminho}",
            f"Tema: {tit}"
        ]
        if desc_limpa:
            partes.append(f"Definição: {desc_limpa}")
        if conhecimento:
            partes.append(f"Institutos correlatos: {conhecimento}")
            
        texto_enriquecido = ". ".join(partes)
        
        item = {
            "tema_id": tid,
            "disciplina": disc,
            "nivel": nivel,
            "titulo": tit,
            "caminho": caminho,
            "descricao_limpa": desc_limpa,
            "conhecimento": conhecimento,
            "texto_enriquecido": texto_enriquecido
        }
        linhas_enriquecidas.append(item)
        
    csv_path = os.path.join(ARTEFATOS_DIR, "temas_enriquecidos.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "tema_id", "disciplina", "nivel", "titulo", "caminho", "descricao_limpa", "conhecimento", "texto_enriquecido"
        ])
        writer.writeheader()
        for row in linhas_enriquecidas:
            writer.writerow(row)
            
    print(f"[OK] Temas enriquecidos gerados em {csv_path} ({len(linhas_enriquecidas)} temas)")
    return linhas_enriquecidas


def gerar_cache_vetorial(
    questoes: List[Dict[str, Any]],
    temas_enriquecidos: List[Dict[str, Any]],
    model_name: str = "intfloat/multilingual-e5-base"
):
    """Gera matrizes densas e esparsas (BM25/TF-IDF) usando texto_completo e salva embeddings_cache.npz."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sentence_transformers import SentenceTransformer
    
    print(f"=== [EMBEDDINGS] Gerando representações densas via {model_name} ===")
    
    q_ids = [q["id"] for q in questoes]
    q_textos = [f"query: {q['texto_completo']}" for q in questoes]
    
    temas_pub = [t for t in temas_enriquecidos if t["nivel"] in (1, 2) and t["tema_id"] not in ("5078", "5234")]
    t_ids = [t["tema_id"] for t in temas_pub]
    t_textos = [f"passage: {t['texto_enriquecido']}" for t in temas_pub]
    
    inativos = [t for t in temas_enriquecidos if t["tema_id"] in ("5078", "5234")]
    inativos_ids = [t["tema_id"] for t in inativos]
    inativos_textos = [f"passage: {t['texto_enriquecido']}" for t in inativos] if inativos else []
    
    print(f"Carregando SentenceTransformer: {model_name}...")
    model = SentenceTransformer(model_name)
    
    print(f"Codificando {len(q_textos)} questões (query: texto_completo)...")
    q_dense = model.encode(q_textos, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    print(f"Codificando {len(t_textos)} temas publicáveis (passage: texto_enriquecido)...")
    t_dense = model.encode(t_textos, batch_size=32, show_progress_bar=True, normalize_embeddings=True)
    
    if inativos_textos:
        print(f"Codificando {len(inativos_textos)} temas inativos para OOD...")
        inat_dense = model.encode(inativos_textos, batch_size=16, show_progress_bar=False, normalize_embeddings=True)
    else:
        inat_dense = np.zeros((0, q_dense.shape[1]), dtype=np.float32)
        
    print("=== [SPARSE] Calculando similaridade esparsa lexical (TF-IDF sublinear) ===")
    # Raw texts without prefix for lexical TF-IDF
    q_raw = [q["texto_completo"] for q in questoes]
    t_raw = [t["texto_enriquecido"] for t in temas_pub]
    inat_raw = [t["texto_enriquecido"] for t in inativos] if inativos else []
    
    tfidf = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.85,
        sublinear_tf=True
    )
    tfidf.fit(t_raw + inat_raw)
    
    q_sparse = tfidf.transform(q_raw)
    t_sparse = tfidf.transform(t_raw)
    
    sim_sparse = (q_sparse @ t_sparse.T).toarray().astype(np.float32)
    
    if len(inat_raw) > 0:
        inat_sparse = tfidf.transform(inat_raw)
        inat_sim_sparse = (q_sparse @ inat_sparse.T).toarray().astype(np.float32)
    else:
        inat_sim_sparse = np.zeros((len(q_ids), 0), dtype=np.float32)
        
    out_npz = os.path.join(ARTEFATOS_DIR, "embeddings_cache.npz")
    np.savez_compressed(
        out_npz,
        questoes_ids=np.array(q_ids),
        questoes_dense=q_dense.astype(np.float32),
        temas_ids=np.array(t_ids),
        temas_dense=t_dense.astype(np.float32),
        inativos_ids=np.array(inativos_ids),
        inativos_dense=inat_dense.astype(np.float32),
        sparse_sim=sim_sparse,
        inativos_sparse_sim=inat_sim_sparse,
        model_name=np.array([model_name])
    )
    
    print(f"[OK] Artefato {out_npz} gerado com sucesso!")
    print(f"     - questoes_dense: {q_dense.shape}")
    print(f"     - temas_dense: {t_dense.shape}")
    print(f"     - sparse_sim: {sim_sparse.shape}")


if __name__ == "__main__":
    print("=== [GERAR ARTEFATOS] Executando processo offline ===")
    questoes = carregar_e_processar_questoes()
    arvore = carregar_e_estruturar_arvore()
    temas_enriq = enriquecer_temas(arvore)
    gerar_cache_vetorial(questoes, temas_enriq)
    print("=== [GERAR ARTEFATOS] Concluído com Sucesso! ===")
