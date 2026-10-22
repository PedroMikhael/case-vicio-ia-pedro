# Relatório Técnico-Científico: Classificação e Decisão Bayesiana em Árvore Taxonômica Jurídica

---

## Sumário Executivo

O presente projeto implementa a arquitetura de classificação e triagem probabilística de questões jurídicas para a taxonomia penal da plataforma Vício, operando com inferência determinística em tempo sub-segundo em CPU padrão e **zero chamadas a Modelos de Linguagem em tempo de execução**.

### Principais Indicadores de Desempenho e Validação (Política Oficial: $\tau = 0,92$ + Portão B)

| Indicador Metodológico | Estimativa Pontual | Intervalo de Confiança (95%) | Base de Cálculo e Protocolo |
| :--- | :---: | :---: | :--- |
| **Precisão Observada na Validação (CV 5x5)** | **100,00%** (19/19) | **[83,18%, 100,00%]** | Wilson Score ($N_{\text{efetivo}} = 19$ itens únicos publicados) |
| **Precisão Prevista em Produção** | **94,00%** (0,94) | **[83,18%, 100,00%]** | Desconto honesto frente à média $\text{Beta}(20,1) = 95,24\%$ e viés post-hoc |
| **Precisão na Validação (Bootstrap)** | 100,00% | [100,00%, 100,00%]* | *Degenerado por zero erros em $k=n=19$; Wilson e Beta são os informativos |
| **Cobertura no Lote Não-Rotulado ($N = 559$)** | **15,56%** (87/559) | — | Taxa de publicação estritamente fora da folha-verdade (0,16) |
| **Cobertura na Folha-Verdade ($N = 100$)** | **19,00%** (19/100) | **[12,51%, 27,78%]** | Wilson Score ($N = 100$ rótulos da Folha-Verdade) |
| **Cobertura no Lote Total ($N = 659$)** | **16,08%** (106/659) | — | Cobertura global (106 publicar, 543 revisar, 10 fora) |
| **Calibração de Probabilidade (Brier Score)** | **0,1857** | — | Brier Score quadrático médio sobre os 100 itens avaliados |
| **Expected Calibration Error (ECE)** | **5,66%** (0,0566) | — | Erro de calibração ponderado em 6 faixas de probabilidade |
| **Fila de Revisão Docente (Lote Total)** | **82,40%** (543/659) | — | Encaminhamento para curadoria com sugestão $n^*$ e parecer |
| **Itens Fora do Catálogo (Lote Total)** | **1,52%** (10/659) | — | Barramento curricular estrito de Direito Penal Militar |
| **Custo Computacional Nuvem (50k q/sem)** | **~R$ 0,06 / lote** | — | Etapa 1 estimada (GPU ~45s) + Etapa 2 medida (CPU ~20s) |
| **Custo Humano da Curadoria (50k q/sem)** | **~400 a 600 h/sem** | — | ~R$ 15k a 32k/sem (~42.000 q triadas; ~5-6 FTEs Nível 2) |
| **Throughput de Decisão em CPU (Medido)** | **~2.514 questões/s** | — | 0,40 ms/item single-thread local (`artefatos/benchmark_resultado.json`) |

---

## a) Arquitetura do Sistema e Seleção de Hiperparâmetros

### 1. Representação Textual Integrada e Scoring Híbrido Multi-Canal
O exame empírico dos enunciados (caso $Q0003$) demonstrou que grande parte das bancas omite o instituto controvertido no texto-base (*"Assinale a opção correta"*), concentrando a lide técnica nas opções de resposta. O módulo `preparar.py` unifica a representação contínua de cada item: `[Enunciado] + "\n\nComando: " + [Comando] + "\n\nAlternativas:\n" + [A, B, C, D, E]`, acompanhada da decodificação de entidades HTML e higienização de escapes.

A similaridade $S(x, t)$ entre a questão $x$ e cada tema publicável $t \in \mathcal{T}_{\text{pub}}$ ($|\mathcal{T}_{\text{pub}}| = 371$) combina os canais denso e léxico:
$$S(x, t) = w_{\text{dense}} \cdot S_{\text{dense}}(x, t) + w_{\text{sparse}} \cdot S_{\text{sparse}}(x, t)$$
* **Proveniência de `temas_enriquecidos.csv`:** Estruturado offline a partir de ementas e dispositivos do Código Penal (CP), Código de Processo Penal (CPP), Leis Especiais (11.343, 11.340, 8.072, 7.210, 12.850, 9.613) e doutrina de Criminologia (Anomia, Labelling, Vitimologia), sem qualquer uso dos textos de teste (zero contaminação).
* **Canal Denso ($S_{\text{dense}}$, peso $w_{\text{dense}} = 0,55$):** Cosseno via `intfloat/multilingual-e5-base` (768d) com prefixos assimétricos (`query:` na questão e `passage:` no catálogo de temas enriquecido).
* **Canal Esparso ($S_{\text{sparse}}$, peso $w_{\text{sparse}} = 0,45$):** TF-IDF com frequência sublinear (`sublinear_tf=True`) e n-gramas $(1, 2)$ para ancoragem lexical de termos unívocos (*"exceção da verdade"*, *"arrependimento posterior"*).
* **Canais Descartados no Scoring:** Expressões regulares rígidas de leis ($w_{\text{symb}} = 0,0$) e prior de disciplina de origem ($w_{\text{origem}} = 0,0$) foram fixados em zero. O prior de origem gerava ruído porque os metadados de sites externos frequentemente confundem Direito Penal com Processual Penal ou Legislação Especial, distorcendo questões interdisciplinares.

### 2. Validação Cruzada Aninhada (Nested CV) e Teoria da Decisão Bayesiana
Hiperparâmetros foram otimizados via **Nested 5-Fold CV** sobre a folha-verdade, minimizando o Brier Score no treino interno (80 itens/fold): $w_{\text{dense}} = 0,55, w_{\text{sparse}} = 0,45, T = 0,016$, obtendo Brier de treino **0,1859** e out-of-fold **0,1857**. A temperatura $T = 0,016$ calibra a faixa estreita de cossenos em probabilidades posteriores nítidas sobre a árvore via Softmax: $p(y = t \mid x) \propto \exp(S(x, t) / T)$.

O edital considera correta a classificação que recai no tema central verdadeiro, nos extras verdadeiros, ou no nó pai ou nós filhos de qualquer um deles. Definimos o **Domínio de Acerto** de um nó publicável $n \in \mathcal{T}_{\text{pub}}$ como:
$$\mathcal{A}(n) = \{n\} \cup \{p \in \mathcal{T}_{\text{pub}} : p = \text{pai}(n), \text{nível}(p) \ge 1\} \cup \{c \in \mathcal{T}_{\text{pub}} : c \in \text{filhos}(n)\}$$
O sistema seleciona o nó $n^*$ que maximiza a probabilidade acumulada no domínio:
$$n^* = \arg\max_{n \in \mathcal{T}_{\text{pub}}} P(\text{acerto} \mid n, x) = \arg\max_{n \in \mathcal{T}_{\text{pub}}} \sum_{t \in \mathcal{A}(n)} p(y = t \mid x)$$
Quando há incerteza entre subtemas irmãos, a probabilidade individual se fragmenta, mas o tema-pai acumula a soma de todas as folhas descendentes, garantindo recuo seguro.

### 3. Política Operacional Oficial: $\tau = 0,92$ + Portão B
1. **Filtro Curricular de Escopo (Estágio 1 - $P(\text{fora})$):**
   * *Regra Militar:* Inspeciona os campos `disciplina_origem` e `assunto_origem`. Ao detectar marcadores de jurisdição militar (`"militar"`, `"cpm"`, `"cppm"`, `"ipm"`), fixa $P(\text{fora}) = 0,95 \ge \tau_{\text{fora}}$ ($0,40$), decidindo `fora` com `tema_central_id = ""` e parecer motivado. Diferente de $w_{\text{origem}} = 0$ (que tentava pontuar temas internos), aqui a origem atua como **filtro negativo de barramento OOD**. *Risco de Generalização:* Em importações futuras que omitam esses campos, a regra não dispara, exigindo detector léxico de CPM no corpo do texto.
   * *Temas Inativos Proibidos:* Compara similaridade contra temas inativos (`5078` e `5234`) e ativos. No lote avaliado de 659 questões, essa regra disparou **exatamente 0 vezes**, atuando como salvaguarda preventiva passiva.
2. **Avaliação da Régua de Publicação com Portão B (Estágio 2):**
   * Se $P(\text{fora}) < \tau_{\text{fora}}$, avalia se $P(\text{acerto} \mid n^*, x) \ge \tau_{\text{pub}}$ ($0,92$) **E** se há concordância de disciplina raiz entre canal denso e esparso ($\text{raiz}_{\text{dense}} == \text{raiz}_{\text{sparse}}$).
   * Em caso afirmativo, emite decisão `publicar` em $n^*$, extraindo `extras_ids` com probabilidade $\ge 0,12$.
3. **Encaminhamento para Curadoria (Estágio 3):**
   * Caso contrário, emite decisão `revisar`. Se a questão atingiu $P^* \ge 0,92$ mas divergiu na raiz, emite motivo específico: *"Concorrência interdisciplinar entre canal denso (...) e esparso (...)"*. Nos demais casos, reporta a dúvida limítrofe entre o 1º e 2º colocados.

---

## b) Protocolo Experimental, Calibração, Métodos Descartados e Sensibilidade

### 1. Métodos Testados e Descartados (Estudo de Ablação Experimental)
O processo de pesquisa avaliou múltiplas hipóteses arquiteturais descartadas formalmente após validação na folha-verdade:

| Configuração / Método Avaliado | Precisão Folha | Cobertura nas 559 | Diagnóstico Experimental e Motivo do Descarte |
| :--- | :---: | :---: | :--- |
| **Baseline 0 (Dense puro / Sem árvore / Sem OOD)** | 32,00% | 100,00% | Argmax direto em classes desbalanceadas; violação massiva da régua de 95%. |
| **1-NN Ponderado (Leave-one-out na Folha)** | 23,00% | 100,00% | Escassez amostral severa (0,27 itens/classe nos 100 rótulos); vizinhos ruidosos. |
| **Centróides Médios por Tema** | 95,00% | 19,00% | Colapso da variância intra-classe; penaliza subtemas com terminologia diversa. |
| **Portão A (Match Exato Denso == Esparso na Folha)** | 100,00% | 4,65% (26/559) | Excessivamente restritivo; descarta concordâncias válidas no tema-pai. |
| **Portão C (Limiar Mínimo Esparso $S_{\text{sparse}} \ge 0,20$)**| 95,65% | 18,60% (104/559)| Ineficaz; o erro $Q0313$ atinge $S_{\text{sparse}} = 0,28$, vazando como falso positivo. |
| **Regras Rígidas Regex ($w_{\text{symb}}$) e Origem ($w_{\text{origem}}$)**| — | — | Pesos fixados em 0,0 por gerarem ruído em certames interdisciplinares. |
| **Iteração 2 (Híbrido Denso+Esparso + Árvore, $\tau = 0,92$)** | 95,65% (22/23) | 18,96% (106/559)| Permite vazamento do erro $Q0313$ ($P=0,9908$); $P(\ge 95\%)$ cai para ~33,9%. |
| **Portão B Oficial ($\tau = 0,92$ + Concordância Raiz)** | **100,00% (19/19)**| **15,56% (87/559)**| **Adotado Oficialmente: Elimina $Q0313$, eleva $P(\ge 95\%)$ para ~64,2% e vence em utilidade.** |

### 2. Tabela de Calibração Empírica por Faixas de Probabilidade ($N = 100$ Folha-Verdade)
O cálculo do Expected Calibration Error (ECE = 5,66%) baseia-se na calibração estrita por faixas:

| Faixa de Probabilidade | Itens ($N$) | Confiança Média | Acurácia Observada | Gap de Calibração | Comportamento Operacional |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **$[0,00; 0,50)$** | 45 | 26,8% | 33,3% | 6,5% | Incerteza difusa; triagem em lote para curadoria. |
| **$[0,50; 0,70)$** | 15 | 59,3% | 60,0% | 0,7% | Disputa entre ramos; encaminhado para Nível 2. |
| **$[0,70; 0,85)$** | 13 | 77,8% | 76,9% | 0,8% | Calibração quase perfeita; retido preventivamente. |
| **$[0,85; 0,90)$** | 4 | 89,0% | 50,0% | 39,0% | **Faixa Crítica de Transição:** 50% de erro; veda publicação. |
| **$[0,90; 0,92)$** | 0 | 0,0% | 0,0% | 0,0% | **Faixa Vazia na Folha:** Sem suporte empírico rotulado. |
| **$[0,92; 1,00]$ (Portão B)** | **19** | **97,4%** | **100,0% (19/19)** | **2,6%** | **Zona Oficial de Publicação Autônoma (Zero Erros).** |

*(Nota: Sem o Portão B, a faixa $[0,92; 1,00]$ abrange 23 itens com acurácia de 95,7% e gap de 1,5%).*

### 3. Sensibilidade do Portão B e Avaliação de $\tau = 0,90 + B$
* **Análise de Sensibilidade:** Pela métrica de utilidade esperada $P(\text{passar}) \times \text{cobertura}$, a política de Portão B supera a política sem portão sempre que a probabilidade real de aprovação for $> 49,4\% \approx 50\%$ (contra $\tau = 0,90$ sem portão, onde $35,8\% \times 21,5\% \approx 7,7$) ou $> 41,3\% \approx 41\%$ (contra $\tau = 0,92$ sem portão, onde $33,9\% \times 19,0\% \approx 6,4$). Assim, mesmo aplicando um desconto punitivo sobre os 64,2% teóricos decorrente do viés de seleção *post-hoc*, o Portão B é **preferível sob esta métrica**.
* **Avaliação de $\tau = 0,90 + B$:** Na folha de 100 itens, $\tau = 0,90 + B$ e $\tau = 0,92 + B$ produzem os **mesmos 19 itens (19/19 acertos, 100%)** porque a faixa $[0,90; 0,92)$ tem zero itens na folha. Nas 559 não-rotuladas, $\tau = 0,90 + B$ publicaria 99 itens ($17,71\%$), atingindo utilidade teórica de ~11,4.
* **Por que a utilidade manda no portão, mas a cautela manda no $\tau$:** A régua de $\ge 95\%$ possui penalidade eliminatória estrita. Como o lote de teste apresenta shift textual (textos 21% mais curtos e comandos não destacados) e a faixa imediatamente anterior $[0,85; 0,90)$ exibiu 50% de erro na folha, os 12 itens não-rotulados na faixa $[0,90; 0,92)$ representam incerteza não validada. A adoção de $\tau = 0,92$ atua como uma margem de cautela que recusa publicar nessa faixa sem evidência empírica direta, protegendo a plataforma.

---

## c) Auditoria dos Dados, Análise de Shift e Diagnóstico de Erros

### 1. Shift de Distribuição e Cobertura Perdida
* **Shift de Distribuição e Cobertura Efetiva:** O lote recente ($Q0360$-$Q0659$) possui textos 21% mais curtos (755 vs 959 caracteres), comando fundido ao enunciado em 100% dos casos e concentração de 10 questões militares (CPM/CPPM). Sob a política oficial:
  - Antigas ($Q0001$-$Q0359$): **55 publicadas em 359 (15,32%)**, 304 revisar, 0 fora.
  - Novas ($Q0360$-$Q0659$): **51 publicadas em 300 (17,00%)**, 239 revisar, 10 fora (3,33%).
  - Lote Não-Rotulado ($N = 559$): **87 publicadas em 559 (15,56% $\approx$ 15,6%)**.
  - Total Geral ($N = 659$): **106 publicadas em 659 (16,08%)**.
* **Cobertura Perdida:** Dos 81 itens retidos em revisão na folha-verdade, **39 questões tiveram seu tema Top-1 correto**, mas não atingiram os critérios de publicação. A acurácia do grupo rejeitado foi de **$39/81 \approx 48,1\%$**, comprovando a segregação efetiva da zona de incerteza.

### 2. Tratamento dos 8 Itens da Folha e Resolução das 10 Questões Militares
* **Itens Fora da Taxonomia na Folha ($N = 8$):** As questões $Q0002, Q0017, Q0039, Q0043, Q0087, Q0252, Q0291, Q0303$ são matérias penais genuínas que sofreram com lacunas cadastrais da árvore (como os nós sem descrição *"Parte 01 e 02"*). **Zero contaminação foi observada na publicação**: todas as 8 foram retidas em `revisar` ($P \in [0,26; 0,90]$), sendo oferecidas no Memorando Pedagógico como base empírica de melhoria.
* **Eliminação dos Falsos Positivos Militares:** As 10 questões militares recentes ($Q0393, Q0416, Q0424, Q0467, Q0479, Q0527, Q0599, Q0649, Q0650, Q0654$) versam sobre CPM/CPPM. O filtro curricular do Estágio 1 fixa $P(\text{fora}) = 0,95$, direcionando todas as 10 para **`fora`** em `publicacao.csv` com `tema_central_id = ""` e parecer motivado.
* **Consolidação Final no CSV (659 questões):** **106 `publicar`**, **543 `revisar`** e **10 `fora`**.

---

## d) Engenharia, Custos e Operação em Produção (50.000 questões/semana)

### 1. Dimensionamento de Custos Computacionais e Latência (Etapa 2 Medida; Etapa 1 Estimada)
A infraestrutura para 50.000 questões semanais foi aferida localmente com o script oficial `src/benchmark.py` (hardware registrado: CPU Intel x86_64, Windows, limitado a `OMP_NUM_THREADS=1` via BLAS, Python 3.13 / NumPy 2.4):
* **Etapa 1 (Embeddings Densos - Estimada):** Em GPU (AWS `g4dn.xlarge` NVIDIA T4 @ US$ 0,526/h), throughput estimado de ~1.100 q/s em FP16 batch dura **~45 segundos** $\implies$ **~R$ 0,04/semana**. Em CPU pura (`c6i.xlarge` a US$ 0,17/h), taxa de ~35 q/s dura **~24 minutos** $\implies$ **~R$ 0,39 a R$ 0,75/semana**.
* **Etapa 2 (Decisão Matricial e Árvore - Medida em Benchmark Local Single-Thread):** Throughput medido de **~2.514 questões/s** (0,398 ms/item em memória, salvo em `artefatos/benchmark_resultado.json`). A inferência e decisão de 50.000 questões executam em **apenas ~19,9 segundos**, custando **< R$ 0,02/semana**. *Nota Metodológica:* A Etapa 2 mede a decisão com matrizes pré-computadas em cache; não inclui vetorização TF-IDF/E5 de textos novos. De ponta a ponta (carga de disco, parse e gravação do CSV), a taxa é de **~646 a 757 q/s** (0,87 a 1,02 s para 659 itens).
* **Custo Total da Infraestrutura:** **~R$ 0,06 por lote semanal** no pipeline híbrido (~R$ 3,12/ano) ou **~R$ 0,75/semana** em CPU pura (~R$ 39,00/ano).

### 2. Dimensionamento do Esforço e Custo Humano da Curadoria (O Custo Dominante)
Com ~16,1% de publicação automática, cerca de **41.950 questões semanais (~83,9%)** alimentam a curadoria docente, configurando o componente financeiramente dominante da operação:
* **Nível 1 (Quase-Certeza, $0,80 \le P < 0,92$, ~4.450 q/sem):** Validação rápida (15 a 20 s/item) $\implies$ **~20 a 25 h/semana** (~0,5 FTE de assistente pedagógico).
* **Nível 2 (Disputa Dogmática, $0,50 \le P < 0,80$, ~11.150 q/sem):** Leitura cuidadosa por especialistas (1 a 1,5 min/item) $\implies$ **~185 a 280 h/semana** (**~5 a 6 professores especialistas equivalentes em tempo integral**).
* **Nível 3 (Incerteza Difusa, $P < 0,50$, ~26.350 q/sem):** Triagem em lote via clusters (30 s/item) $\implies$ **~220 a 330 h/semana** (~5 a 8 FTEs de assistentes).
* **Consolidado da Operação Humana:** Demanda total de **~425 a 635 horas semanais** (equipe de 11 a 15 profissionais). A R$ 35–50/h pedagógica, a esteira representa **~R$ 15.000 a R$ 32.000 semanais** (~R$ 780k a 1,6 milhão/ano), confirmando que a automação economiza milhares de horas humanas de alta especialização.

### 3. Ciclo de Retreino Semanal e Protocolo de Auditoria Amostral
* **Ciclo de Retreino:** Temperatura $T$, limiares $\tau_{\text{pub}}$ e $\tau_{\text{fora}}$, pesos $w_{\text{dense}}, w_{\text{sparse}}$ e vocabulário TF-IDF. A correção de viés utiliza amostragem estratificada inversa: auditoria cega amostra alta confiança a $\pi_{\text{audit}} = 200 / 8.050 \approx 2,48\%$, enquanto itens revisados amostram a fila a $\pi_{\text{rev}}$. Ponderar cada observação por $w_i = 1/\pi_i$ restaura a distribuição marginal sem distorcer probabilidades.
* **Protocolo de Auditoria Semanal ($N = 200$ itens auditados):**
  - **Teste de Hipótese Unilateral ($H_0: p \ge 0,95$ com $\alpha = 0,05$):** Rejeição formal de $H_0$ ocorre se, e somente se, o limite superior de Wilson cair abaixo de 0,95, o que se dá estritamente com **$k \le 184$ acertos em 200** ($\hat{p} \le 92,0\%$, $p$-valor binomial $0,0444 < 0,05$).
  - **Alarme Preventivo (Sinal Amarelo — $185 \le k \le 188$ acertos):** Sob precisão real de 94,0%, probabilidade de disparar sozinho é de **$\approx 39,1\%$** (probabilidade cumulativa $k \le 188$ é $\approx 54,3\%$), disparando inspeção contínua sem interromper a esteira.
  - **Parada Crítica (Sinal Vermelho — $k \le 184$ acertos):** Probabilidade de disparar é de **$\approx 15,2\%$** sob precisão de 94%. Paralisa imediatamente publicações automáticas para recalibração.

---

## e) Memorando Pedagógico para o Professor Curador de Direito Penal e Processual Penal

**DE:** Coordenação de Inteligência Aplicada e Avaliação Educacional  
**PARA:** Professor Curador de Direito Penal e Direito Processual Penal  
**ASSUNTO:** Critérios Dogmáticos de Classificação e Solicitações de Alinhamento Curricular  

Prezado Professor,

A integridade do plano de estudos depende da consistência conceitual do acervo. Para mitigar o risco pedagógico de oferecer questões desajustadas aos estudantes, adotamos uma política de conservadorismo: perante sobreposição de institutos ou incerteza dogmática razoável, o sistema recusa a publicação automática e transfere o item para curadoria humana acompanhado de parecer preliminar.

Com o objetivo de aprimorar continuamente o catálogo da plataforma, submetemos a Vossa Senhoria **cinco solicitações operacionais**:

### 1. Dimensionamento Docente de Nível 2 e Nomeação de Sênior para Auditoria
A esteira de Nível 2 (Disputa Dogmática, ~11.150 questões/semana) demanda **~5 a 6 professores especialistas equivalentes em tempo integral (FTEs)** para validação técnica cuidadosa das alternativas. Solicitamos a aprovação desse dimensionamento e a indicação formal de um docente sênior para liderar a auditoria amostral cega semanal ($N = 200$).

### 2. Tratamento da Dualidade Ação Penal (5466/5475) vs Efeitos da Condenação (5209)
O caso $Q0313$ evidenciou concorrência entre *Efeitos da Condenação* (CP art. 91, I, tema `5209`) e *Ação Civil Ex Delicto* (CPP arts. 63-68, tema `5475`, filho de `5466`). O modelo atual barrou a publicação por divergência interdisciplinar (Portão B). Recomendamos registrar diretriz definindo se questões relativas a título executivo reparatório civil devem concentrar-se na disciplina substantiva ou na processual.

### 3. Definição sobre Ramos Homônimos Duplicados
Identificamos temas homônimos cadastrados em duplicidade sob disciplinas distintas: *Tráfico de Drogas* (ID `4708` Legislação Especial vs `6728` Penal), *Lavagem de Dinheiro* (`4702` vs `6723`), *Lei Maria da Penha* (`4704` vs `6724`) e *Abuso de Autoridade* (`4684` vs `6716`). Solicitamos definir se a plataforma adotará o critério da especialidade ou a alocação no Código Penal, unificando os IDs.

### 4. Correção de Lacunas Taxonômicas da Folha-Verdade e Bloco 6700 ("Partes 01 e 02")
Identificamos 8 questões padrão-ouro da folha ($Q0002, Q0017, Q0039, Q0043, Q0087, Q0252, Q0291, Q0303$) que versam sobre Penal/Processo Penal, mas foram rotuladas como `fora_da_taxonomia` devido a lacunas cadastrais: ausência de nó para *Teorias da Pena em Criminologia* ($Q0002$) e fragmentação arbitrária de tópicos em nós sem descrição ementária (*Partes 01 e 02* na faixa `6716` a `6738`). Disponibilizamos essas 8 questões como base empírica para consolidação dos nós.

### 5. Diretriz Curricular Definitiva para Questões de Direito Militar
Identificamos 10 questões recentes de certames militares versando sobre CPM e CPPM. Como o catálogo não possui ramos militares, direcionamos preventivamente todas as 10 para a decisão `fora`. Solicitamos definir se a plataforma criará ramos de Direito Militar ou se tais concursos devem ser formalmente arquivados como fora de escopo.

Colocamo-nos à inteira disposição para colaborar no aperfeiçoamento contínuo da taxonomia da plataforma.
