# Guia dos Experimentos — em português simples

> Os detalhes técnicos e o "porquê" de cada decisão estão no [ADR.md](ADR.md)
> (ADR-017 a ADR-020). Aqui está a explicação.

**Não leia isto do começo ao fim.** Vá direto ao que você precisa:

| Se você quer… | Vá para |
|---|---|
| Entender o essencial em 1 minuto | [Se você só decorar cinco frases](#se-você-só-decorar-cinco-frases) (no fim) |
| Saber o que cada experimento provou | [Os 4 experimentos](#os-4-experimentos) |
| Entender um termo que viu numa tela | [Glossário sem jargão](#glossário-sem-jargão) |
| Saber onde olhar no app | [O que cada tela responde](#o-que-cada-tela-do-app-responde) |
| Se preparar para a apresentação | [Perguntas da banca](#perguntas-da-banca-com-resposta-pronta) e [O que NÃO foi provado](#o-que-não-foi-provado) |

---

## A ideia central, em um parágrafo

Antes, o projeto **construía** um recomendador de treino. Agora ele **mede** um
recomendador — e o resultado da medição foi que a regra atual tem um defeito
concreto e identificável. Isso parece ruim, mas é o contrário: um recomendador
testado que revelou um defeito é uma contribuição de TCC mais forte que um
recomendador nunca testado. A pergunta que você responde na banca deixa de ser
*"eu construí isso"* e passa a ser *"eu construí isso, medi, e descobri o que
está errado"*.

---

## Os 4 experimentos

| # | Pergunta | Resposta encontrada |
|---|----------|---------------------|
| 1 | As métricas dos modelos são confiáveis? | **Não.** Eram otimistas, porque o mesmo atleta estava em treino e em teste. |
| 2 | A regra de recomendação é melhor que o trivial? | **Não.** Perde de "sempre recomende leve". |
| 3 | Treinar mais/melhor melhora o desempenho na prova? | **Sim.** O treino explica 62% da variação no tempo de meia maratona. |
| 4 | A meta de 60% leve está certa? | **Está impossível.** O classificador não consegue produzir 60% de leve. |

Os experimentos 1 e 2 diagnosticam; 3 e 4 explicam a causa.

---

## 1. O protocolo estava errado

**O problema.** Todos os 8 modelos dividiam os dados sorteando **corridas** para
treino e teste. Como um atleta tem centenas de corridas, o mesmo atleta caía nos
dois lados. O modelo então "reconhecia" atletas que já tinha visto, em vez de
generalizar para um atleta novo — que é exatamente o caso de uso real (alguém
conecta o Strava pela primeira vez e pede uma análise).

**A correção.** Dividir por **atleta**: se o atleta 42 está no treino, nenhuma
corrida dele aparece no teste.

**O que mudou nos números.** No Random Forest de pace, o erro médio (MAE) sai de
**0,751** para **0,840** min/km — o erro real é ~12% maior do que se reportava.

Mas o achado importante é outro. Medindo o MAE em cada uma das 5 partições
separadamente, a **variação entre elas** vai de 0,010 (protocolo antigo) para
**0,043** (protocolo correto) — mais de 4 vezes maior. Ou seja: **o modelo acerta
muito em alguns atletas e erra muito em outros**, e o protocolo antigo escondia
isso completamente, porque com o mesmo atleta nos dois lados todas as partições
ficavam parecidas.

**Uma conclusão antiga caiu.** O projeto afirmava que Random Forest era melhor que
Regressão Linear para prever pace. Com o teste estatístico correto, a diferença
entre os dois **não existe** (`p = 0,399`). O que realmente vence é o `RF
Regressor` em pace e o XGBoost em tempo — esses sim com diferença real.

---

## 2. A regra de recomendação (meta global) não funcionava

> **Atualização (ADR-024):** a recomendação de produto passou a usar **só**
> déficit vs meta do **cluster**. Os números abaixo são do protocolo antigo
> com a `META` global (60/20/12/8), mantidos como histórico do porquê ela
> foi abandonada.

Coloquei a regra por **déficit vs meta global** para competir com recomendadores
propositalmente bobos, no mesmo teste: dado o histórico do atleta até hoje, qual
será o tipo do próximo treino dele?

| Recomendador | Acerto | F1-macro |
|---|---|---|
| Sorteia da distribuição do próprio atleta | 35% | **0,341** |
| Repete o último treino | 32% | 0,316 |
| Sorteio uniforme (chute puro) | 24% | 0,233 |
| **Déficit vs meta global** (abandonada) | 26% | **0,142** |
| Sempre "leve" | 34% | 0,128 |

A regra global ficava em penúltimo. Ela perdia de "sempre leve" com significância
estatística (`p = 0,0007`) e **empatava com o chute puro** (`p = 1,0`).

**Por quê?** Porque ela recomendava `leve` em **78%** dos casos. Era praticamente
a constante "leve" disfarçada de regra. A causa está no experimento 4.

**A ablação** (remover um componente por vez) mostrou o que cada parte da regra
contribui:

| Componente | Efeito de removê-lo |
|---|---|
| `PESOS_DEFICIT` (os pesos 1,1 e 1,2) | +0,0008 — **inerte, não faz nada** |
| Regra anti-fadiga | +0,005 — remover **melhora** |
| Janela de 6 semanas → 2 semanas | +0,028 — 2 semanas é **melhor** |
| Janela de 6 → 12 semanas | −0,014 — 12 semanas é pior |

Três conclusões acionáveis: os pesos podem ser apagados, a regra anti-fadiga
atrapalha um pouco, e a janela deveria ser mais curta.

> ⚠️ **Cuidado ao falar disso na banca.** Este teste mede **imitação** — se o
> sistema acerta o que o atleta *fez*. Ele **não** mede se o conselho é *bom*: um
> atleta que treina mal é imitado com sucesso por um sistema que recomenda mal.
> Quem mede qualidade é o experimento 3.

---

## 3. O treino prediz o desempenho

Este é o experimento que dá utilidade ao projeto, e ele precisou de uma **segunda
base de dados** ([long-distance-running-dataset](https://www.kaggle.com/datasets/mexwell/long-distance-running-dataset)),
porque a base original não tem resultado de prova nenhum.

**O desfecho.** Para 6.821 atletas, extraí o melhor tempo de **meia maratona**
(21 km) e o treino das 12 semanas anteriores. Mediana de 1h41, com 10% abaixo de
1h25 e 10% acima de 2h09 — faixa amadora plausível, o que é o teste de sanidade
do dado.

**O resultado.** O treino explica **62%** da variação no tempo de prova. As
associações mais fortes:

| O que se mede no treino | Associação com o tempo de prova |
|---|---|
| Pace mediano dos treinos | **+0,77** (treina devagar → prova lenta) |
| Volume semanal | −0,40 (treina mais → prova rápida) |
| Nº de treinos | −0,30 |
| Proporção de treino leve | −0,10 (mais leve → prova rápida) |
| Proporção de intervalado | +0,15 (mais tiro → prova lenta) |

As duas últimas linhas importam para o TCC: **mais corrida leve associa-se a
prova mais rápida**, e mais intervalado ao contrário. É a direção que a `META`
supõe, e coincide com a literatura de treino polarizado (a "regra 80/20").

**E a composição de treino importa mesmo?** Testei se saber *como* o atleta
distribui os treinos ajuda a prever a prova, **além** de já saber quanto e quão
rápido ele treina. Sim, ajuda — e o ganho é estatisticamente real
(`p = 0,00000002`), embora pequeno (o R² vai de 0,617 para 0,633). Repeti com
Random Forest e a conclusão se manteve, então não é artefato do modelo linear.

**Tradução:** a premissa da regra (equilibrar a distribuição de treinos faz
diferença) **tem suporte nos dados**. O problema não é a ideia. É o número.

---

## 4. A meta de 60% leve é impossível

Aqui está a causa raiz de tudo.

A `META` diz que o atleta ideal faz **60% de treino leve**. Fui medir quanto de
leve os atletas realmente fazem:

| Tipo | Meta pede | Atletas reais fazem (mediana) | Faixa observada (p5 a p95) |
|---|---|---|---|
| **leve** | **60%** | **28%** | **11% a 48%** ❌ fora |
| longão | 20% | 27% | 13% a 43% ✅ dentro |
| ritmo | 12% | 22% | 6% a 44% ✅ dentro |
| intervalado | 8% | 19% | 4% a 42% ✅ dentro |

A meta de `leve` está **acima de 95% dos atletas** da amostra: o percentil 95 é
48%, e a meta pede 60%.

**Por que é impossível, e não apenas raro.** O classificador de tipo de treino
(ADR-004) é *relativo ao atleta*: `longao` é definido como toda corrida acima do
**percentil 75 de distância daquele atleta**. Por definição matemática, isso
coloca ~25% das corridas de *todo* atleta na classe `longao`. Sobram ~75% para
dividir entre leve, ritmo e intervalado, e as faixas de pace dividem isso de forma
aproximadamente equilibrada. Não existe como chegar a 60% de leve.

**A consequência em cadeia:**

```
meta pede 60% de leve
   └─> mas o classificador só produz ~30%
        └─> o déficit de leve é positivo para 111 de 111 atletas
             └─> a regra recomenda leve em 78% dos casos
                  └─> vira a constante "leve" e perde do baseline
```

**E por que eu não simplesmente troquei os números?** Porque tentei, e não deu
para confiar no resultado. Otimizei a meta contra o tempo de 21 km e obtive
valores diferentes dependendo do tamanho da amostra: com 1.686 atletas a meta
aprendida ganhava da atual; com 6.821 ela perdia. A margem de erro de um dos
componentes (`longão`) vai de 13% a 42% — mais larga que a própria correção que
ela propõe. Trocar `60/20/12/8` por outros quatro números seria substituir um
chute por outro chute com aparência de precisão.

**O que ficou provado, então:** a *direção* da meta está certa (mais leve, menos
intervalado — experimento 3 confirma), e o *nível* de leve está num valor
inalcançável. A correção certa é fazer a meta respeitar o que o classificador
consegue produzir — não inventar novos números.

---

## O que cada tela do app responde

| Tela | Pergunta que ela responde | Se você só olhar uma coisa |
|---|---|---|
| 🔬 Validação da recomendação · topo | A regra funciona? | A caixa verde **Conclusão em 30 segundos** |
| 🔬 Validação · aba 1 (vs baselines) | A regra por cluster bate o trivial? | Gráfico de barras + **`rec_leve_%`** + matriz de confusão |
| 🔬 Validação · aba 2 | Quais partes da regra importam? | A coluna **`delta_f1_vs_completa`**: verde = remover melhorou |
| 🔬 Validação · aba 3 | Os números dos modelos auxiliares são confiáveis? | A coluna **`otimismo`** |
| 🏃 Recomendação de Treino | Qual o próximo treino do atleta (por cluster)? | Ficha do cluster + bloco **Próximo treino recomendado** |
| 📈 Comparação de Modelos | (métricas antigas, mantidas por paridade com o notebook) | O aviso amarelo no topo |

> Desfecho de prova / meta aprendida saíram do menu Streamlit (ADR-023); os
> experimentos continuam via `python -m dataset.experimentos desfecho|meta`.

As telas são densas de propósito — elas são a evidência. Para apresentar, use a
caixa de conclusão de cada aba e deixe as tabelas nos expanders.

---

## Glossário sem jargão

Cada termo aqui aparece nas telas. A coluna que importa é a última.

**Bootstrap (intervalo de confiança).**
Sorteio com reposição, repetido 2.000 vezes, para descobrir o quanto uma métrica
oscilaria se eu tivesse coletado outra amostra de atletas.
*Por que precisei:* dizer "o MAE é 0,84" sem margem de erro não permite saber se
0,84 e 0,79 são diferentes de verdade.
*Sem isso:* eu declararia vencedores com base em diferenças que podem ser ruído.

**Bootstrap "de atletas" (e não de corridas).**
No sorteio, eu reamostro **atletas inteiros**, não corridas individuais.
*Por que precisei:* as corridas de um mesmo atleta são parecidas entre si — elas
não são observações independentes. Reamostrar corridas fingiria que eu tenho
40.000 informações independentes quando tenho ~114 (os atletas).
*Sem isso:* os intervalos de confiança sairiam estreitos demais, e eu afirmaria
ter certeza que não tenho.

**Teste de Wilcoxon pareado.**
Compara dois modelos olhando o erro **do mesmo atleta** nos dois, e pergunta se um
é consistentemente melhor.
*Por que "pareado":* alguns atletas são simplesmente difíceis de prever, e essa
variação entre atletas é muito maior que a diferença entre modelos. Comparar
pareado cancela a dificuldade do atleta.
*Por que "Wilcoxon" e não teste t:* o teste t supõe que os erros seguem uma curva
normal, e erros de predição não seguem.

**Correção de Holm.**
Ajusta os `p` quando eu faço muitos testes de uma vez.
*Por que precisei:* comparei 8 modelos entre si e 7 recomendadores entre si. Quem
faz 28 testes com 5% de chance de erro em cada acaba encontrando um "resultado"
falso quase por garantia.
*Sem isso:* eu reportaria descobertas que são sorte.

**Modelos encaixados (*nested*).**
O modelo B tem todas as features do A, mais as que eu quero testar. Se B não for
melhor que A, as features novas não carregam informação.
*Por que precisei:* é a única forma justa de perguntar "a composição de treino
importa **além** do volume?". Comparar dois modelos com features diferentes não
responderia isso.

**Polinômio de Scheffé.**
Uma forma de modelo usada quando as variáveis são **proporções que somam 100%**
(o caso das 4 proporções de treino).
*Por que precisei:* minha primeira tentativa usou um modelo linear comum, e ele
respondeu "faça 100% de treino leve". Não foi um achado — é uma propriedade
matemática: um modelo linear sobre proporções sempre põe o ótimo num extremo.
Scheffé acrescenta os termos de interação entre pares, o que permite um ótimo
equilibrado.
*Sem isso:* eu teria concluído "treine só leve", que é falso.

**GroupKFold.**
Divide os dados em 5 partes garantindo que um mesmo atleta nunca apareça em duas.
*É a correção central do experimento 1.*

---

## Perguntas da banca, com resposta pronta

**"Por que a tela de Comparação de Modelos mostra números diferentes da tela de
Protocolo?"**
De propósito. A primeira reproduz o notebook original, com o protocolo antigo; a
segunda usa o protocolo corrigido. Mantive as duas lado a lado, com a diferença
rotulada, em vez de trocar os números em silêncio — a diferença entre elas *é* um
dos resultados do trabalho.

**"Então o seu recomendador não funciona?"**
A regra atual, como está, não supera um baseline trivial — e eu demonstrei isso
com um protocolo estatístico explícito. Também identifiquei a causa exata: a meta
de 60% de treino leve é matematicamente inalcançável dado o classificador. E
mostrei que a premissa por trás dela está certa, porque mais treino leve
associa-se a prova mais rápida em 6.821 atletas de uma base independente. O
trabalho entrega o diagnóstico e o caminho de correção.

**"Por que você não corrigiu a meta?"**
Porque eu não consegui estimá-la com precisão suficiente. A margem de erro de um
dos componentes é mais larga que a correção proposta, e o resultado se inverte
conforme o tamanho da amostra. Trocar quatro números afirmados por quatro números
igualmente incertos não seria uma melhoria, seria só outra afirmação. O que os
dados sustentam é que a meta precisa respeitar a faixa que o classificador produz.

**"Por que usar uma segunda base de dados?"**
Porque a base original não tem nenhum resultado de prova. Sem desfecho, a
recomendação só pode ser avaliada contra a própria premissa, o que é circular. A
segunda base permite extrair o melhor tempo de meia maratona e perguntar se o
treino se associa a desempenho melhor.

**"Por que meia maratona e não maratona?"**
Porque o público-alvo do projeto é o corredor amador, e 21 km é a distância de
prova dominante nesse público. Otimizar para maratona calibraria o sistema para
um perfil que não é o do usuário.

**"Como você sabe que o tempo de prova extraído é confiável?"**
Dois testes de sanidade. Primeiro, a distribuição dos tempos é plausível para
amadores (mediana 1h41). Segundo, na variante de maratona eu cruzei a coluna de
provas declaradas com as datas oficiais das World Marathon Majors, e a ordem entre
as provas reproduz a realidade: Boston, que exige índice de qualificação, é a mais
rápida; Chicago e Nova York, por sorteio, as mais lentas.

---

## O que NÃO foi provado

Diga isso antes que perguntem — é o que separa um trabalho honesto de um
exagerado.

1. **Nada de causalidade.** Todos os dados são observacionais. Mais treino leve
   *associa-se* a prova mais rápida; não está provado que um atleta que passe a
   correr mais leve vá melhorar. Atletas mais rápidos podem treinar diferente
   *porque* já são rápidos. Provar causalidade exigiria um experimento controlado.

2. **O desfecho é um proxy.** O "melhor tempo de 21 km" pode ter sido um treino
   forte, não uma prova oficial. É capacidade demonstrada, não resultado de prova.

3. **A avaliação da recomendação mede imitação.** Acertar o próximo treino do
   atleta não é o mesmo que dar bom conselho.

4. **O RF de tipo de treino tem alvo circular.** Ele prevê `tipo_treino` a partir
   das mesmas quantidades que *definem* `tipo_treino`. O F1 de 0,91 dele não é
   evidência de capacidade preditiva, e isso está sinalizado na própria tela.

5. **A meta ótima não foi identificada.** Só o diagnóstico de que a atual é
   inalcançável.

---

## Se você só decorar cinco frases

1. O protocolo antigo inflava as métricas porque o mesmo atleta estava em treino e
   teste; corrigi dividindo por atleta.
2. A regra de recomendação atual perde de "sempre recomende leve", porque emite
   `leve` em 78% dos casos.
3. A causa é que a meta de 60% de leve é matematicamente inalcançável: o
   classificador de longão consome ~25% das corridas por definição.
4. A ideia por trás da meta está certa — em 6.821 atletas de uma base
   independente, mais treino leve associa-se a prova de 21 km mais rápida.
5. Não troquei os números da meta porque a estimativa não é confiável; entreguei o
   diagnóstico e o caminho, não um chute novo.
