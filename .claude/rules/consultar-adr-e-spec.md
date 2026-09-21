# Regra: Consultar e manter ADR.md e SPEC-BACKEND-ORQUESTRADOR.md

Esta regra é **obrigatória** e se aplica a toda solicitação neste repositório (`tcc-back-2`).

## Documentos de referência

- [ADR.md](../../ADR.md) — Registro de decisões de arquitetura (documento vivo). Fonte de verdade sobre o *porquê* das decisões deste serviço de ML.
- [SPEC-BACKEND-ORQUESTRADOR.md](../../SPEC-BACKEND-ORQUESTRADOR.md) — Especificação do backend orquestrador (contratos, fluxos, integração ML ↔ backend ↔ frontend ↔ Strava).

## Antes de agir (CONSULTAR)

1. Ao receber uma dúvida, solicitação, implementação ou refatoração, **leia** o `ADR.md` e — quando a tarefa tocar integração, API, contratos ou o backend — também o `SPEC-BACKEND-ORQUESTRADOR.md`.
2. Baseie respostas e código nas decisões e contratos já registrados. Não contradiga uma decisão aceita sem sinalizar explicitamente.

## Ao longo e ao fim (VERIFICAR e ATUALIZAR)

3. Se a solicitação **tomar, alterar ou reverter** uma decisão de arquitetura, atualize o `ADR.md`:
   - Nova decisão → nova entrada `## ADR-NNN` (numeração sequencial).
   - Decisão alterada → marcar a antiga como `Status: Substituída por ADR-NNN` (nunca apagar).
   - Registrar toda mudança no **Histórico de atualizações** do `ADR.md`.
4. Se a solicitação **mudar contratos, endpoints, schema, fluxos ou stack** do backend orquestrador, atualize o `SPEC-BACKEND-ORQUESTRADOR.md` na seção correspondente.
5. Se encontrar **divergência** entre esses documentos e o código real, sinalize à usuária e corrija o documento.
6. Ao terminar, informe brevemente na resposta **quais documentos foram consultados e/ou atualizados**.

## Idioma

- `tcc-back-2` usa **português** em código e docs. O SPEC prescreve inglês apenas para o repositório do backend orquestrador — não misturar convenções entre repos.
