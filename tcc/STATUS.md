# STATUS — redação do TCC

Formato adotado: **monografia** (modelo do TCC-2 de referência), salvo decisão em contrário
da autora (ver `PENDENCIAS.md`).

Ordem de redação adotada: capítulo 3 primeiro (é onde estão os fatos), depois 2, 1, 4,
pré-textuais e apêndices.

## Próximo passo

**Escrever a seção 3.2.2 — Limpeza e pré-processamento.** Base: `dataset/preprocessamento.py`,
`dataset/constants.py`, ADR-003, ADR-001, ADR-002, ADR-018 (tolerância a FC ausente),
`tests/test_preprocessamento_sem_fc.py`, FLUXO §2. Números de referência já apurados:
42.116 → 40.534 corridas e 116 → 114 atletas (`01-FATOS.md` F02, F06, F21, F22). A primeira
equação numerada do trabalho, a do *pace*, entra nessa seção como equação (1). Próximos
números livres: Figura 4, Quadro 6, Tabela 2.

## Checklist por seção

### Capítulo 3 — Desenvolvimento (`capitulos/03-desenvolvimento.md`)

- [x] Abertura do capítulo 3 (299 palavras)
- [ ] 3.1 ESPECIFICAÇÃO — **pendente**; exige decidir o conjunto de RF/RNF a partir do que o repositório faz de fato. Reservados Quadro 1 (RF) e Quadro 2 (RNF).
- [x] 3.2 DESCRIÇÃO DO MODELO — abertura e visão geral do fluxo (2.280 palavras)
- [x] 3.2.1 Obtenção dos dados (5.766 palavras)
- [ ] 3.2.2 Limpeza e pré-processamento ← **próximo**
- [ ] 3.2.3 Classificação do tipo de treino
- [ ] 3.2.4 Perfil do atleta por regras
- [ ] 3.2.5 Clusterização de corredores
- [ ] 3.2.6 Modelos de previsão de pace e de tempo
- [ ] 3.2.7 Recomendação por déficit em relação à meta do cluster
- [ ] 3.2.8 Recomendação via modelo de linguagem
- [ ] 3.2.9 Protocolo de validação
- [ ] 3.2.10 Resultados e análise
- [ ] 3.2.11 Evolução do projeto e achados negativos
- [ ] 3.2.12 Ameaças à validade e limitações
- [ ] 3.3 OPERACIONALIDADE DA APLICAÇÃO

### Demais capítulos

- [ ] 1 Introdução
- [ ] 2 Fundamentação teórica (incl. 2.8 Trabalhos correlatos)
- [ ] 4 Conclusões
- [ ] Pré-textuais
- [ ] Referências
- [ ] Apêndices
- [ ] Consolidação em `tcc-completo.md`

## Documentos de apoio

- [x] `01-FATOS.md` — criado, cobrindo a **coleta de dados** (fatos F01–F45, conflitos C01–C05). Deve ser estendido a cada nova seção.
- [x] `PLANO-DE-FIGURAS.md` — criado, com Figuras 1–3, Quadros 1–5 (1 e 2 reservados) e Tabela 1.
- [x] `PENDENCIAS.md` — criado.
- [ ] `00-PLANO.md` — não criado nesta invocação (Fase 2 parcial: o esqueleto de títulos está no próprio arquivo do capítulo 3).
- [ ] `REFERENCIAS.md` — não criado; nenhuma obra externa foi citada até aqui.

## Contagem de palavras

| Capítulo / seção | Palavras (sem blocos de código) | Mínimo | Situação |
|---|---|---|---|
| Abertura do cap. 3 | 299 | — | ok |
| 3.2 abertura | 2.280 | 900 | ok |
| 3.2.1 | 5.766 | 1.200 | ok |
| **Total do cap. 3 até aqui** | **8.475** | 9.000 (capítulo inteiro) | em andamento |

Observação: as contagens incluem o texto dos marcadores `[INSERIR …]`, que serão
substituídos pelas figuras e não contam como texto final. O corpo redigido efetivo é
menor; a margem sobre os mínimos continua folgada.
