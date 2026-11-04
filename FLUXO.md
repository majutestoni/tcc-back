# TCC — Fluxo completo: do Strava à validação da recomendação

Este repositório (`tcc-back-2`) é o **serviço de Machine Learning** do TCC: recebe histórico de corridas, limpa, classifica, agrupa corredores, recomenda o próximo treino e **mede se a regra faz sentido**.

Documentos irmãos (não substituem este texto):

| Arquivo | Papel |
|---------|--------|
| [ADR.md](ADR.md) | Registro formal do *porquê* de cada decisão (ADR-001…026) |
| [GUIA-EXPERIMENTOS.md](GUIA-EXPERIMENTOS.md) | Conclusões em português simples + perguntas de banca |
| [SPEC-BACKEND-ORQUESTRADOR.md](SPEC-BACKEND-ORQUESTRADOR.md) | Como o backend vai conectar Strava ↔ este ML ↔ frontend |

---

## Mapa mental (o fluxo inteiro)

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  COLETA                                                                 │
│  Hoje: CSV Kaggle (export Strava).  Futuro: OAuth Strava via backend. │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  LIMPEZA + ENGENHARIA                                                   │
│  gênero → FC → filtros → pace → tipo_treino relativo ao atleta         │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  CATEGORIZAÇÃO                                                          │
│  Perfil por regras (iniciante…avançado)  +  KMeans por gênero           │
│  → perfil_cluster: volume_alto | volume_moderado                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  MODELOS AUXILIARES (pace / tempo)                                      │
│  LinReg, RF, XGBoost — avaliados com GroupKFold por atleta              │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  RECOMENDAÇÃO (produto)                                                 │
│  A) Déficit vs meta do cluster (regras)                                 │
│  B) Gemini (LLM) sobre as mesmas features determinísticas               │
└───────────────────────────────┬─────────────────────────────────────────┘
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  VALIDAÇÃO                                                              │
│  1) Eficácia por desfecho (tempo de 21 km)  ← veredito                  │
│  2) Teste antigo (adivinhar hábito Strava)  ← F1 baixo não é nota       │
│  3) Protocolo dos 8 modelos + ablação + meta aprendida                  │
└─────────────────────────────────────────────────────────────────────────┘
```

No ecossistema completo (produto):

```text
Usuário → Frontend (run-metrics-mate)
              ↕ REST
         Backend orquestrador  ←→  Strava (OAuth + sync)
              ↕ HTTP (futuro POST /analyze)
         Este repo (ML)
              ↕
         Banco (tokens, atividades, resultados)
```

Hoje a demo roda **só** neste repo, com CSV local e Streamlit. A API HTTP ainda é proposta (ADR-009).

---

## 1. Coleta dos dados (Strava)

### O que existe hoje neste repo

Não há OAuth Strava aqui. A base **primária** é um dump público do Kaggle:

- [running-races-strava](https://www.kaggle.com/datasets/olegoaer/running-races-strava)
- Arquivo versionado: `dataset/raw-data-kaggle.csv` (separador `;`)
- Cada linha ≈ uma corrida de um atleta (distância, tempo, FC, gênero, timestamp, …)

**Por quê CSV e não API ao vivo?** O TCC precisa de volume histórico estável e reprodutível para treinar, clusterizar e validar. A integração Strava em produção fica no backend orquestrador (SPEC).

### Fonte de validação externa (desfecho de prova)

A base Kaggle **não tem** resultado de meia/maratona. Para medir se treinar “melhor” se associa a correr mais rápido, usamos:

- [long-distance-running-dataset](https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset) (~2,3 GB, **não versionada**)
- Download: `bash scripts/baixar_long_distance.sh`
- Adaptador: `dataset/fontes/long_distance.py` (schema canônico, pode não ter FC)

**Por quê segunda fonte?** Sem variável de desfecho, só dá para medir se o sistema *imita* o hábito do Strava — e imitar hábito ruim não prova que o conselho é bom (ADR-018).

---

## 2. Limpeza e preprocessamento

Entrada: CSV bruto. Saída: DataFrame Polars pronto para modelos.

**Por quê Polars?** API de expressões e performance no pipeline (ADR-001).  
**Por quê limiares só em `constants.py`?** Evita strings mágicas espalhadas e facilita calibração (ADR-002).

Pipeline fixo em `dataset/preprocessamento.py` → `preprocessar(df)` (ADR-003), nesta ordem:

| Passo | O que faz | Por quê |
|-------|-----------|---------|
| 1. Gênero válido | Mantém só `M` / `F` | Remove ruído; KMeans por gênero depende disso |
| 2. Imputar FC | Ausentes → **mediana** | Muitos treinos sem monitor; mediana é robusta a outliers |
| 3. Corridas válidas | Distância ≥ 1 km; pace entre 2 e 15 min/km; FC entre 40 e 220 | Descarta GPS quebrado, pausas mal registradas, FC absurda |
| 4. Pace | `pace_min_km = (tempo/60) / (dist/1000)` | Unidade comum para modelos e recomendação |
| 5. Tipo de treino | Ver seção seguinte | Rótulo usado pela recomendação e pelos experimentos |

Sem FC útil na fonte (long-distance), o pipeline detecta e adapta: `intervalado` usa só pace (ADR-018).

---

## 3. Classificação do tipo de treino (relativa ao atleta)

Cada corrida vira um de: **`longao` · `intervalado` · `leve` · `ritmo`**.

**Por quê relativo e não absoluto?** “Longo” para quem corre 5 km ≠ “longo” para quem corre 20 km. Usar percentis e desvios **por atleta** (`.over(athlete)`) torna a etiqueta justa (ADR-004).

Regras (prioridade nesta ordem):

1. **Longão** — distância ≥ percentil 75 do próprio atleta  
2. **Intervalado** — pace bem abaixo da média do atleta (× desvio) e, se houver FC, FC ≥ mediana  
3. **Leve** — pace bem acima da média  
4. **Ritmo** — pace perto da média  
5. Fallback → `leve`

Efeito colateral importante (medido depois): como ~25% das corridas caem em longão *por construção* (p75), a distribuição real fica perto de ~33% leve / 25% longão / 24% ritmo / 18% intervalado — **nunca** 60% leve. Isso derrubou a meta global antiga (ver §8).

---

## 4. Categorização de corredores

Duas lentes complementares (ADR-005/006), a partir de agregação mediana por atleta (`montar_perfil_atletas`). Exige histórico mínimo (`MIN_CORRIDAS_PERFIL = 20` no perfil por regras).

### 4.1 Perfil por regras (interpretável)

Eixos combinados → `iniciante` | `intermediario` | `fundista` | `avancado`.

**Por quê regras além do cluster?** A banca e o usuário precisam de rótulo em português; centroide de KMeans sozinho não explica.

### 4.2 KMeans por gênero

- Features padrão: **distância + pace**  
- `StandardScaler` antes do KMeans (`random_state=42`)  
- Homens e mulheres separados → rótulos `C{i}-M` / `C{i}-F`  
- Validação interna: silhouette, Davies-Bouldin, Calinski-Harabasz; estabilidade; grid de k  

**Por quê por gênero?** Escalas e distribuições diferem; misturar distorce os clusters (ADR-006).

### 4.3 Nomeação → `perfil_cluster`

Os clusters viram chaves de produto:

| Chave | Ideia |
|-------|--------|
| `volume_alto` | Mais volume/distância no grupo |
| `volume_moderado` | Volume mais baixo (também é o **fallback** se não houver cluster) |

Código: `dataset/categorizar/nomear_clusters.py` + uso em `recomendacao_cluster.py`.

---

## 5. Modelos auxiliares (pace e tempo)

Não são o produto principal, mas fazem parte do TCC e da tela **Comparação de Modelos**:

- Previsão de **pace** (LinReg, RF, RF Regressor, …)  
- Previsão de **tempo** (RF, LinReg, XGBoost, …)  
- Contrato único de saída: `ResultadoModelo` (ADR-008)

**Avaliação correta (ADR-017):** `GroupKFold` **por atleta** — o mesmo corredor nunca fica em treino e teste. Protocolo antigo (split por corrida) era otimista (~12% a menos de erro em pace) e escondia variação entre atletas.

> A tela Comparação de Modelos **mantém de propósito** o protocolo antigo (paridade com notebooks). Os números defensáveis estão na **Validação da recomendação** e na CLI `dataset.experimentos`.

---

## 6. Como funciona cada recomendação

Há **duas** recomendações de produto e **uma** histórica. As três usam a janela das últimas **6 semanas** (`SEMANAS_JANELA`).

### 6.A Recomendação por cluster (regras) — caminho principal

Arquivos: `dataset/models/previsao_treino/recomendacao_cluster.py` + `recomendar_proximo_treino` em `model.py`.

**Passo a passo:**

1. Descobrir o `perfil_cluster` do atleta (`volume_alto` ou `volume_moderado`).  
2. Pegar a **meta** desse perfil (`META_POR_PERFIL`):

   | Perfil | leve | longão | ritmo | intervalado |
   |--------|------|--------|-------|-------------|
   | `volume_alto` | 45% | 28% | 17% | 10% |
   | `volume_moderado` | 42% | 22% | 22% | 14% |

3. Calcular a **distribuição atual** de tipos na janela de 6 semanas.  
4. **Déficit** = meta − atual (por tipo).  
5. Escolher o tipo com maior **déficit ponderado** (`PESOS_DEFICIT`: intervalado 1,2; longão 1,1; resto 1,0).  
6. **Anti-fadiga (ADR-026):**  
   - se ≥ 2 treinos duros (intervalado/longão) nos últimos 3 → força `leve`;  
   - se o último foi intervalado e a regra pediria outro duro → `leve`;  
   - se o último foi longão e pediria outro longão → `leve`.  
7. **Distância:** mediana calibrada do *cluster* misturada com a mediana do *atleta* naquele tipo (55% / 45%); longão sobe no máx. ~12% além do típico do atleta.  
8. **Pace alvo:** pace médio do atleta + offset do tipo (leve mais lento, intervalado mais rápido).  
9. **Tempo estimado:** distância × pace.  
10. Texto humano: `explicar_recomendacao(...)`.

**Por quê meta por cluster e não uma meta global?**  
A meta antiga `60/20/12/8` (ADR-007) era **inalcançável** com o classificador relativo: déficit de `leve` positivo em 111/111 atletas e ~76–78% das recomendações eram `leve`. No teste de imitação, perdia de “sempre leve” e empatava com o chute. Por isso ADR-024/026: metas **dentro da região factível** (~leve 42–45%).

**Por quê regras e não só ML?** Explicabilidade para banca e usuário; débito e anti-fadiga auditáveis. Existe RF de recomendação no código, mas está **desativado** no `model_main`.

### 6.B Recomendação via Gemini (LLM) — complementar

Pacote `gemini/`.

**Ideia:** o código faz o trabalho determinístico; o LLM só redige o conselho.

1. `montar_features_atleta` monta JSON com: perfil por regras, n corridas, pace/distância/FC, distribuição na janela, **meta do cluster**, déficits, últimos treinos.  
2. Prompt de “treinador” + temperatura baixa (~0,3).  
3. Saída **estruturada** (Pydantic): `tipo`, `descricao`, `distancia_km`, `pace_alvo_min_km`, `tempo_estimado_min`, `justificativa`, `alertas`.  
4. Persistência opcional em `output/recomendacoes.csv` (histórico da demo).

**Por quê não deixar o LLM inventar as features?** Menos alucinação, menos tokens, mesma base numérica da regra (ADR-010). Gemini **não substitui** a regra; enriquece a justificativa.

CLI: `python -m gemini.recomendacao <athlete_id>` (precisa `GEMINI_API_KEY` no `.env`).

### 6.C Meta global (ADR-007) — só histórico

`META = 60% leve / 20% longão / 12% ritmo / 8% intervalado` ainda aparece em `model.py` para:

- experimentos de desfecho/meta aprendida (ADR-018/019);  
- paridade com notebooks na tela Comparação de Modelos.

**Não** é mais o caminho da demo de recomendação nem dos baselines de produto.

---

## 7. Demo Streamlit (o que o usuário vê)

```bash
bash run.sh
# ou
streamlit run app/Home.py
```

| Tela | Função no fluxo |
|------|-----------------|
| **Fluxo do sistema** | Mapa do pipeline + tomada de decisão (este documento, navegável) |
| Visualização do Dataset | EDA após limpeza |
| Comparação de Modelos | Pace/tempo (protocolo antigo, rotulado) |
| Categorização | KMeans, validações, fichas |
| Atletas | Tabela filtrável |
| Recomendação de Treino | Cluster + Gemini lado a lado |
| Histórico | CSV das gerações da IA |
| **Validação da recomendação** | *A regra funciona?* |

---

## 8. Validações — o fim do fluxo (e o núcleo científico)

### 8.1 Protocolo dos modelos (ADR-017)

- Partição por **atleta**, IC por bootstrap, Wilcoxon pareado + Holm.  
- Achado: RF ≠ LinReg em pace (`p_holm ≈ 0,4`); o protocolo antigo mentia otimismo.

### 8.2 Imitação vs baselines (contraste, não veredito)

Pergunta: “dado o histórico até hoje, o sistema acerta o **próximo tipo que o atleta fez**?”

Isso mede **adivinhar o hábito no Strava**, não qualidade do conselho. Na UI fica na **aba 2** (*Por que F1 baixo não importa*).

Ablação (remover peças da regra): pesos quase inertes; anti-fadiga antiga atrapalhava um pouco a imitação; janela de 2 semanas imitava melhor que 6 — mas imitar ≠ treinar bem.

### 8.3 Desfecho de prova (ADR-018) — utilidade

Na fonte externa: melhor tempo de **21 km** (e maratona como robustez).

Achado: composição/volume de treino **explica** grande parte da variação do tempo de meia (ordem de ~62% no guia). Ou seja: faz sentido recomendar distribuição de treinos.

### 8.4 Meta aprendida (ADR-019) — diagnóstico

Tentativa de aprender a meta ótima via Scheffé a partir do desfecho:

- método e diagnóstico **aceitos**: 60% leve está fora da região factível;  
- números ótimos **rejeitados**: ICs largos, vencedor instável — não dá para cravar quatro porcentagens “ideais” e jogar na produção.

Por isso as metas de produto foram **recalibradas à mão** para a faixa alcançável (ADR-026), não copiadas do otimizador.

### 8.5 Eficácia por desfecho (ADR-026) — veredito da UI

Módulo: `dataset/experimentos/eficacia_desfecho.py`.

Ideia: atletas mais **próximos da meta do cluster** tendem a tempos melhores de 21 km? Correlação distância-à-meta × tempo; tempos por tercil de aderência.

Na tela **Validação da recomendação**, a **aba 1** é o veredito; a aba 2 explica por que F1 baixo não importa.

CLI:

```bash
python -m dataset.experimentos protocolo
python -m dataset.experimentos baselines
bash scripts/baixar_long_distance.sh
python -m dataset.experimentos desfecho --max-atletas 12000
python -m dataset.experimentos meta --max-atletas 12000
python -m dataset.experimentos tudo
pytest tests/
```

---

## 9. Por que cada escolha grande (resumo)

| Escolha | Por quê |
|---------|---------|
| Polars + `constants.py` | Performance e uma fonte de verdade para colunas/limiares |
| Tipo de treino relativo | “Longo/rápido” só faz sentido no histórico da pessoa |
| KMeans por gênero + regras | Cluster encontra estrutura; regras explicam |
| Meta por cluster (não 60% leve) | Meta global era matematicamente inalcançável e viciava em `leve` |
| Déficit ponderado + anti-fadiga | Corrige desequilíbrio sem empilhar carga dura |
| Distância misturada cluster×atleta | Plano do grupo, dose individual |
| Gemini só no texto | Features confiáveis; justificativa rica |
| GroupKFold por atleta | Caso de uso real = atleta novo |
| Segunda base com 21 km | Validar utilidade, não só imitação |
| Streamlit + notebooks | Demo para banca; notebooks para exploração |
| Sem API ainda | Foco científico neste repo; HTTP no SPEC |

---

## 10. Setup rápido

```bash
# macOS / Linux
bash run.sh

# Windows
.\run.ps1
```

Manual:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # GEMINI_API_KEY se for usar LLM
streamlit run app/Home.py
python -m dataset.models.model_main
```

Após mudar metas/recomendação, limpe `.streamlit/cache/` se a UI parecer desatualizada.

---

## Estrutura do código (onde achar cada etapa)

```text
dataset/
  raw-data-kaggle.csv       # coleta (snapshot Strava/Kaggle)
  preprocessamento.py       # limpeza + tipo_treino
  categorizar/              # perfil, KMeans, nomeação de cluster
  models/previsao_pace/     # modelos auxiliares
  models/previsao_tempo/
  models/previsao_treino/   # regra global (histórico) + cluster (produto)
  fontes/                   # Kaggle + long-distance
  avaliacao/                # protocolo por atleta
  experimentos/             # baselines, desfecho, eficácia, meta
gemini/                     # recomendação LLM
app/                        # demo Streamlit
tests/
scripts/baixar_long_distance.sh
```
