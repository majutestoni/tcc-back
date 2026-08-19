# Especificação: Backend Orquestrador — Run Metrics Mate

**Projeto**: TCC — Sistema de análise e recomendação de treinos para corredores  
**Componente**: Backend orquestrador (novo repositório)  
**Criado em**: 2026-08-10  
**Status**: Draft — pronto para Spec-Driven Development  

> **Como usar com [Spec Kit](https://github.com/github/spec-kit)**
>
> 1. `specify init run-metrics-backend --integration cursor` (ou seu agente preferido)
> 2. `/speckit.constitution` — cole a seção **5. Princípios e Regras Técnicas**
> 3. `/speckit.specify` — cole as seções **1, 2, 3 e 4** (contexto, fluxos, histórias, requisitos)
> 4. `/speckit.plan` — cole a seção **6. Plano Técnico de Implementação**
> 5. `/speckit.tasks` → `/speckit.implement`

---

## 1. Contexto de Negócio

### 1.1 Problema

Corredores amadores utilizam o Strava para registrar treinos, mas não recebem análises personalizadas nem recomendações práticas sobre o próximo treino com base no histórico real. Existe um serviço de Machine Learning (repositório `tcc-back-2`) capaz de:

- preprocessar e analisar dados de corrida;
- prever pace e tempo;
- categorizar corredores (KMeans);
- recomendar o próximo treino (leve, longão, ritmo, intervalado).

Existe também um frontend (`https://github.com/majutestoni/run-metrics-mate`) para exibir métricas ao usuário.

**Falta** um backend que conecte o usuário ao Strava, sincronize atividades, transforme os dados no formato esperado pelo ML e entregue resultados ao frontend.

### 1.2 Objetivo do Backend

Criar um **serviço orquestrador** que:

1. Permita ao usuário **conectar a conta Strava** (sem login próprio do app).
2. Sincronize atividades de corrida do Strava.
3. Envie dados normalizados ao **serviço de ML**.
4. Persista tokens, atividades e resultados de análise.
5. Exponha uma **API REST** consumida exclusivamente pelo frontend.

### 1.3 Escopo do Produto (visão completa)

```text
Usuário → Frontend (run-metrics-mate)
              ↕ REST/JSON
         Backend orquestrador (ESTE PROJETO)
              ↕ OAuth + sync          ↕ HTTP
           Strava API              Serviço ML (tcc-back-2)
              ↕
         Banco de dados
```

### 1.4 Usuário-alvo

- Corredor amador ou intermediário que já usa Strava.
- Conecta a conta Strava uma vez e visualiza dashboard com perfil, métricas e recomendação de treino.
- Não há cadastro, senha ou login no app — apenas **"Conectar com Strava"**.

### 1.5 Fora de Escopo (v1)

- Autenticação própria (email/senha, JWT de usuário, OAuth Google etc.).
- Escrita de atividades no Strava (criar treinos, upload).
- Pagamentos ou planos.
- App mobile nativo.
- Multi-tenant enterprise ou admin panel.

---

## 2. Fluxos do Sistema

### 2.1 Fluxo principal — Primeiro acesso

```mermaid
sequenceDiagram
    actor U as Usuário
    participant FE as Frontend
    participant BE as Backend
    participant ST as Strava API
    participant DB as Banco
    participant ML as Serviço ML

    U->>FE: Acessa aplicação
    FE->>BE: GET /auth/status
    BE-->>FE: { connected: false }
    U->>FE: Clica "Conectar com Strava"
    FE->>BE: GET /auth/strava/login
    BE->>ST: Redirect OAuth (scope read, activity:read)
    ST->>U: Tela de autorização Strava
    U->>ST: Autoriza
    ST->>BE: GET /auth/strava/callback?code=...
    BE->>ST: POST /oauth/token (troca code)
    ST-->>BE: access_token + refresh_token
    BE->>ST: GET /athlete
    ST-->>BE: perfil do atleta
    BE->>DB: Salva sessão + tokens
    BE-->>FE: Redirect + Set-Cookie session_id
    FE->>BE: POST /api/sync
    BE->>ST: GET /athlete/activities
    ST-->>BE: lista de atividades
    BE->>DB: Persiste atividades
    BE->>ML: POST /analyze (payload normalizado)
    ML-->>BE: perfil + cluster + recomendação
    BE->>DB: Salva analysis_cache
    BE-->>FE: { sync: ok, dashboard: {...} }
    FE->>U: Exibe dashboard
```

### 2.2 Fluxo — Acesso recorrente (sessão ativa)

```mermaid
sequenceDiagram
    participant FE as Frontend
    participant BE as Backend
    participant DB as Banco
    participant ML as Serviço ML

    FE->>BE: GET /auth/status (cookie)
    BE->>DB: Busca sessão
    BE-->>FE: { connected: true, athlete: {...} }
    FE->>BE: GET /api/dashboard
    BE->>DB: Busca analysis_cache + activities
    alt cache recente (< 24h)
        BE-->>FE: dashboard do cache
    else cache expirado
        BE->>BE: POST /api/sync (interno)
        BE->>ML: POST /analyze
        BE->>DB: Atualiza cache
        BE-->>FE: dashboard atualizado
    end
```

### 2.3 Fluxo — Refresh de token Strava

```mermaid
sequenceDiagram
    participant BE as Backend
    participant ST as Strava API
    participant DB as Banco

    BE->>DB: Busca tokens da sessão
    alt access_token expirado
        BE->>ST: POST /oauth/token (grant_type=refresh_token)
        ST-->>BE: novos tokens
        BE->>DB: Atualiza tokens
    end
    BE->>ST: Chamada autenticada à API
```

### 2.4 Fluxo — Desconectar Strava

```mermaid
sequenceDiagram
    participant U as Usuário
    participant FE as Frontend
    participant BE as Backend
    participant DB as Banco

    U->>FE: Clica "Desconectar"
    FE->>BE: POST /auth/disconnect
    BE->>DB: Remove tokens e invalida sessão
    BE-->>FE: { connected: false }
    BE-->>FE: Clear-Cookie session_id
```

### 2.5 Pipeline de dados (Backend → ML)

```text
Strava Activity JSON
    → adapter (mapeamento de campos)
    → DataFrame/lista no schema ML
    → POST /analyze no serviço ML
    → resposta JSON estruturada
    → persistência + resposta ao frontend
```

**Schema interno esperado pelo ML** (compatível com `dataset/raw-data-kaggle.csv`):

| Campo ML | Tipo | Origem Strava |
|----------|------|---------------|
| `athlete` | int | `athlete.id` |
| `gender` | `"M"` \| `"F"` | `athlete.sex` |
| `timestamp` | string `"dd/MM/yyyy HH:mm"` | `start_date_local` |
| `distance (m)` | float | `distance` |
| `elapsed time (s)` | float | `elapsed_time` |
| `elevation gain (m)` | float | `total_elevation_gain` (default 0) |
| `average heart rate (bpm)` | float \| null | `average_heartrate` |

Campos derivados (`pace_min_km`, `tipo_treino`, perfil, recomendação) são calculados **pelo serviço ML**, não pelo backend.

---

## 3. Histórias de Usuário e Critérios de Aceite

### US-01 — Conectar conta Strava (Prioridade: P1)

**Como** corredor, **quero** conectar minha conta Strava com um clique, **para** que o sistema acesse meus treinos sem criar conta no app.

**Teste independente**: Clicar em "Conectar Strava", autorizar no Strava, retornar ao app com status `connected: true`.

**Cenários**:

1. **Given** usuário não conectado, **When** clica "Conectar Strava", **Then** é redirecionado à autorização Strava.
2. **Given** usuário autorizou no Strava, **When** callback é processado, **Then** sessão é criada e cookie HTTP-only é definido.
3. **Given** conexão bem-sucedida, **When** frontend consulta `/auth/status`, **Then** retorna `connected: true` com nome/id do atleta.

---

### US-02 — Sincronizar atividades (Prioridade: P1)

**Como** corredor conectado, **quero** que minhas corridas Strava sejam importadas, **para** alimentar análises e recomendações.

**Teste independente**: Após sync, `GET /api/activities` retorna lista de corridas com distância, tempo e data.

**Cenários**:

1. **Given** sessão Strava válida, **When** `POST /api/sync`, **Then** atividades de corrida (`type=Run`) são persistidas.
2. **Given** sync repetido, **When** mesma atividade já existe, **Then** não duplica (upsert por `strava_activity_id`).
3. **Given** token expirado, **When** sync é solicitado, **Then** backend renova token automaticamente e conclui sync.

---

### US-03 — Ver dashboard com análise ML (Prioridade: P1)

**Como** corredor conectado, **quero** ver meu perfil, métricas e recomendação de próximo treino, **para** orientar meu planejamento.

**Teste independente**: `GET /api/dashboard` retorna perfil, cluster (se disponível) e recomendação com tipo, distância e pace alvo.

**Cenários**:

1. **Given** atividades sincronizadas, **When** dashboard é solicitado, **Then** backend chama ML e retorna JSON consolidado.
2. **Given** atleta com histórico insuficiente (< 20 corridas), **When** dashboard é solicitado, **Then** retorna mensagem clara de histórico insuficiente (sem erro 500).
3. **Given** FC ausente em algumas atividades, **When** ML processa, **Then** análise prossegue (ML trata nulls).

---

### US-04 — Desconectar Strava (Prioridade: P2)

**Como** corredor, **quero** desconectar minha conta Strava, **para** revogar o acesso do app aos meus dados.

**Cenários**:

1. **Given** sessão ativa, **When** `POST /auth/disconnect`, **Then** tokens removidos e cookie invalidado.
2. **Given** desconectado, **When** acessa endpoints protegidos, **Then** retorna 401 com mensagem "Strava não conectado".

---

### US-05 — Status de conexão (Prioridade: P2)

**Como** frontend, **quero** saber se o usuário está conectado ao Strava, **para** exibir botão correto (Conectar vs Dashboard).

**Cenários**:

1. **Given** cookie de sessão válido, **When** `GET /auth/status`, **Then** `{ connected: true, athlete: { id, name } }`.
2. **Given** sem cookie ou sessão expirada, **When** `GET /auth/status`, **Then** `{ connected: false }`.

---

### Casos de Borda

- Strava retorna lista vazia de atividades → dashboard informa "sem corridas encontradas".
- Atleta sem `sex` no perfil Strava → usar `"M"` como fallback ou excluir atleta com aviso (documentar decisão).
- Serviço ML indisponível → retornar 503 com `{ error: "ml_unavailable" }` e dados de atividades ainda disponíveis.
- Rate limit Strava (429) → retry com backoff exponencial (máx. 3 tentativas).
- Apenas atividades `Run` são processadas; outros tipos são ignorados na sync.

---

## 4. Requisitos Funcionais

| ID | Requisito |
|----|-----------|
| **FR-001** | O sistema MUST implementar OAuth 2.0 authorization code flow com Strava. |
| **FR-002** | O sistema MUST NOT exigir cadastro, senha ou JWT de usuário próprio. |
| **FR-003** | O sistema MUST identificar o usuário via sessão HTTP-only (cookie) após conexão Strava. |
| **FR-004** | O sistema MUST armazenar `access_token` e `refresh_token` Strava de forma segura (nunca expor ao frontend). |
| **FR-005** | O sistema MUST renovar tokens Strava automaticamente antes de chamadas expiradas. |
| **FR-006** | O sistema MUST sincronizar atividades de corrida (`type=Run`) via `GET /athlete/activities`. |
| **FR-007** | O sistema MUST transformar atividades Strava para o schema ML documentado na seção 2.5. |
| **FR-008** | O sistema MUST chamar o serviço ML via HTTP (`POST /analyze`) e repassar resposta ao frontend. |
| **FR-009** | O sistema MUST expor API REST JSON consumida apenas pelo frontend (CORS configurado). |
| **FR-010** | O sistema MUST persistir atividades, tokens (por sessão/atleta) e cache de análise. |
| **FR-011** | O sistema MUST implementar sync incremental usando parâmetro `after` da API Strava. |
| **FR-012** | O sistema MUST retornar erros HTTP padronizados com corpo JSON `{ "error": "code", "message": "..." }`. |
| **FR-013** | O frontend MUST NOT chamar Strava ou ML diretamente — apenas o backend. |
| **FR-014** | O sistema MUST fornecer endpoint de health check `GET /health`. |

### Entidades Principais

| Entidade | Descrição |
|----------|-----------|
| **Session** | Sessão do usuário pós-OAuth; contém `session_id`, `strava_athlete_id`, timestamps. |
| **StravaToken** | Tokens OAuth vinculados à sessão/atleta; `access_token`, `refresh_token`, `expires_at`. |
| **Activity** | Corrida sincronizada; campos normalizados + `raw_json` opcional. |
| **AnalysisCache** | Resultado ML cacheado; `profile_json`, `recommendation_json`, `computed_at`. |

### Critérios de Sucesso

| ID | Critério |
|----|----------|
| **SC-001** | Usuário conecta Strava e vê dashboard em menos de 60 segundos (incluindo sync inicial). |
| **SC-002** | 100% dos tokens Strava permanecem no backend (nunca aparecem em responses ao frontend). |
| **SC-003** | Sync incremental não duplica atividades existentes. |
| **SC-004** | Dashboard retorna recomendação de treino quando há histórico suficiente (≥ 20 corridas, alinhado ao ML). |
| **SC-005** | API documentada via OpenAPI/Swagger automático (FastAPI `/docs`). |

---

## 5. Princípios e Regras Técnicas (Constitution)

Use esta seção no comando `/speckit.constitution`.

### Princípio 1 — Simplicidade para TCC

Preferir soluções mínimas e funcionais. Evitar microserviços desnecessários, filas, cache distribuído ou GraphQL na v1.

### Princípio 2 — Backend como único ponto de integração

Frontend fala só com backend. Backend fala com Strava e ML. Tokens Strava nunca vão ao browser.

### Princípio 3 — Sem auth própria

Não implementar registro, login, JWT de usuário ou bcrypt. Identificação exclusiva via sessão pós-OAuth Strava.

### Princípio 4 — Contratos explícitos

Toda integração (frontend ↔ backend, backend ↔ ML) MUST ter schemas Pydantic documentados. Responses MUST ser JSON previsível.

### Princípio 5 — Código Python idiomático

Type hints em funções públicas. Separação clara: `routers/`, `services/`, `models/`, `schemas/`. Funções de negócio testáveis sem HTTP.

### Princípio 6 — Configuração por ambiente

Secrets (`STRAVA_CLIENT_SECRET`, `SESSION_SECRET`) via variáveis de ambiente. Nunca commitar `.env`. Fornecer `.env.example`.

### Princípio 7 — Testabilidade

Testes unitários para adapter Strava→ML e serviços de sync. Testes de integração para rotas OAuth (mock Strava).

### Princípio 8 — Observabilidade mínima

Logging estruturado em operações críticas: OAuth callback, sync, chamada ML, refresh token. Sem expor tokens nos logs.

### Princípio 9 — Resiliência pragmática

Retry em 429/503 Strava. Timeout de 30s em chamadas ML. Fallback gracioso quando ML indisponível.

### Princípio 10 — Idioma

Código, nomes de variáveis e commits em **inglês**. Mensagens de erro ao usuário e documentação em **português**.

---

## 6. Plano Técnico de Implementação

Use esta seção no comando `/speckit.plan`.

### 6.1 Stack Tecnológica

| Camada | Tecnologia |
|--------|------------|
| Linguagem | Python 3.11+ |
| Framework HTTP | FastAPI |
| Servidor ASGI | Uvicorn |
| HTTP client (Strava, ML) | httpx (async) |
| Banco de dados | SQLite (dev) / PostgreSQL (prod) |
| ORM | SQLAlchemy 2.x |
| Migrations | Alembic |
| Sessão | Starlette SessionMiddleware (cookie HTTP-only) |
| Validação | Pydantic v2 |
| Config | pydantic-settings |
| Testes | pytest + pytest-asyncio + httpx mock |
| Container | Docker + docker-compose |

### 6.2 Repositórios Relacionados

| Repositório | Papel | Integração |
|-------------|-------|------------|
| `run-metrics-backend` | **Este projeto** (orquestrador) | — |
| `tcc-back-2` | Serviço ML | HTTP `ML_SERVICE_URL` |
| `run-metrics-mate` | Frontend | Consome API do backend |
| Strava API | Fonte de dados | OAuth + REST |

### 6.3 Estrutura de Pastas

```text
run-metrics-backend/
├── app/
│   ├── main.py                 # FastAPI app, CORS, middleware
│   ├── config.py               # Settings (env vars)
│   ├── dependencies.py         # get_session, get_db
│   ├── routers/
│   │   ├── auth.py             # /auth/strava/*
│   │   ├── activities.py       # /api/activities
│   │   ├── dashboard.py        # /api/dashboard
│   │   └── health.py           # /health
│   ├── services/
│   │   ├── strava_client.py    # OAuth + API Strava
│   │   ├── strava_adapter.py   # Strava JSON → schema ML
│   │   ├── ml_client.py        # HTTP → serviço ML
│   │   ├── sync_service.py     # Orquestra sync + ML
│   │   └── session_service.py  # CRUD sessão
│   ├── models/                 # SQLAlchemy models
│   │   ├── session.py
│   │   ├── strava_token.py
│   │   ├── activity.py
│   │   └── analysis_cache.py
│   └── schemas/                # Pydantic DTOs
│       ├── auth.py
│       ├── activity.py
│       ├── dashboard.py
│       └── ml.py
├── alembic/
├── tests/
│   ├── unit/
│   └── integration/
├── .env.example
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

### 6.4 Variáveis de Ambiente

```env
# App
APP_ENV=development
APP_HOST=0.0.0.0
APP_PORT=8000
FRONTEND_URL=http://localhost:5173
SESSION_SECRET=change-me-in-production

# Strava OAuth
STRAVA_CLIENT_ID=
STRAVA_CLIENT_SECRET=
STRAVA_REDIRECT_URI=http://localhost:8000/auth/strava/callback
STRAVA_SCOPES=read,activity:read

# ML Service
ML_SERVICE_URL=http://localhost:8001
ML_SERVICE_TIMEOUT_SECS=30

# Database
DATABASE_URL=sqlite:///./run_metrics.db
# DATABASE_URL=postgresql://user:pass@localhost:5432/run_metrics
```

### 6.5 API REST — Contratos

#### Auth

| Método | Rota | Descrição | Auth |
|--------|------|-----------|------|
| GET | `/auth/strava/login` | Redirect para Strava OAuth | — |
| GET | `/auth/strava/callback` | Callback OAuth; cria sessão | — |
| GET | `/auth/status` | Status da conexão | cookie |
| POST | `/auth/disconnect` | Remove conexão | cookie |

**Response `/auth/status`**:

```json
{
  "connected": true,
  "athlete": {
    "id": 18042525,
    "firstname": "João",
    "lastname": "Silva"
  }
}
```

#### Dados

| Método | Rota | Descrição | Auth |
|--------|------|-----------|------|
| POST | `/api/sync` | Sincroniza Strava + reprocessa ML | cookie |
| GET | `/api/activities` | Lista corridas sincronizadas | cookie |
| GET | `/api/dashboard` | Perfil + recomendação + métricas | cookie |
| GET | `/health` | Health check | — |

**Response `/api/dashboard`**:

```json
{
  "athlete_id": 18042525,
  "synced_at": "2026-08-10T14:00:00Z",
  "activities_count": 45,
  "profile": {
    "pace_mediano": 5.2,
    "distancia_mediana_km": 10.4,
    "frequencia_semana": 2.1,
    "n_corridas": 45
  },
  "cluster": {
    "id": 1,
    "label": "cluster_1"
  },
  "recommendation": {
    "tipo": "longao",
    "descricao": "Longão em ritmo confortável",
    "distancia_km": 14.0,
    "pace_alvo_min_km": 5.65,
    "tempo_estimado_min": 79.1
  },
  "distribution": {
    "leve": 0.55,
    "longao": 0.15,
    "ritmo": 0.20,
    "intervalado": 0.10
  },
  "warnings": []
}
```

**Response erro padronizado**:

```json
{
  "error": "insufficient_history",
  "message": "Histórico insuficiente. São necessárias pelo menos 20 corridas."
}
```

### 6.6 Contrato Backend → ML

**Request** `POST {ML_SERVICE_URL}/analyze`:

```json
{
  "athlete_id": 18042525,
  "gender": "M",
  "activities": [
    {
      "timestamp": "15/12/2019 09:08",
      "distance_m": 2965.8,
      "elapsed_s": 812,
      "elevation_m": 17.4,
      "avg_hr": 150.3
    }
  ]
}
```

**Response esperada do ML**:

```json
{
  "profile": {
    "pace_mediano": 5.2,
    "distancia_mediana_km": 10.4,
    "frequencia_semana": 2.1,
    "n_corridas": 45
  },
  "cluster": { "id": 1, "label": "cluster_1" },
  "recommendation": {
    "tipo": "longao",
    "descricao": "Longão em ritmo confortável",
    "distancia_km": 14.0,
    "pace_alvo_min_km": 5.65,
    "tempo_estimado_min": 79.1
  },
  "distribution": {
    "leve": 0.55,
    "longao": 0.15,
    "ritmo": 0.20,
    "intervalado": 0.10
  }
}
```

> **Nota**: O serviço ML (`tcc-back-2`) ainda precisa expor essa API FastAPI. O backend MUST tratar indisponibilidade do ML graciosamente.

### 6.7 Modelo de Dados (SQLAlchemy)

```sql
sessions (
  id              TEXT PRIMARY KEY,   -- UUID session_id
  strava_athlete_id BIGINT NOT NULL UNIQUE,
  created_at      TIMESTAMP NOT NULL,
  updated_at      TIMESTAMP NOT NULL
)

strava_tokens (
  id              INTEGER PRIMARY KEY,
  session_id      TEXT NOT NULL REFERENCES sessions(id),
  access_token    TEXT NOT NULL,
  refresh_token   TEXT NOT NULL,
  expires_at      TIMESTAMP NOT NULL,
  updated_at      TIMESTAMP NOT NULL
)

activities (
  id                  INTEGER PRIMARY KEY,
  strava_athlete_id   BIGINT NOT NULL,
  strava_activity_id  BIGINT NOT NULL UNIQUE,
  name                TEXT,
  distance_m          FLOAT NOT NULL,
  elapsed_s           FLOAT NOT NULL,
  elevation_m         FLOAT DEFAULT 0,
  avg_hr              FLOAT,
  activity_date       TIMESTAMP NOT NULL,
  raw_json            TEXT,
  synced_at           TIMESTAMP NOT NULL
)

analysis_cache (
  id                  INTEGER PRIMARY KEY,
  strava_athlete_id   BIGINT NOT NULL UNIQUE,
  profile_json        TEXT NOT NULL,
  recommendation_json TEXT NOT NULL,
  cluster_json        TEXT,
  distribution_json   TEXT,
  computed_at         TIMESTAMP NOT NULL
)
```

### 6.8 Integração Strava — Detalhes Técnicos

**OAuth URLs**:

- Authorize: `https://www.strava.com/oauth/authorize`
- Token: `https://www.strava.com/oauth/token`
- API base: `https://www.strava.com/api/v3`

**Parâmetros authorize**:

```text
client_id={STRAVA_CLIENT_ID}
redirect_uri={STRAVA_REDIRECT_URI}
response_type=code
approval_prompt=auto
scope=read,activity:read
```

**Endpoints Strava usados**:

| Endpoint | Uso |
|----------|-----|
| `POST /oauth/token` | Trocar code e refresh token |
| `GET /athlete` | Perfil (id, nome, sex) |
| `GET /athlete/activities` | Listar corridas (`after`, paginação) |

**Filtro de atividades**: processar apenas `type == "Run"`.

**Referência**: https://developers.strava.com/docs/reference/

### 6.9 CORS e Sessão

- CORS: permitir origem `FRONTEND_URL` com `credentials: true`.
- Cookie: `HttpOnly`, `SameSite=Lax`, `Secure` em produção.
- Frontend MUST enviar requests com `credentials: 'include'`.

### 6.10 Cache de Análise

- Reprocessar ML se `computed_at` > 24 horas OU após novo sync com atividades novas.
- Sync MUST ser idempotente (upsert por `strava_activity_id`).

### 6.11 Docker Compose (dev)

```yaml
services:
  backend:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    depends_on: [db]
  db:
    image: postgres:16
    environment:
      POSTGRES_DB: run_metrics
      POSTGRES_USER: app
      POSTGRES_PASSWORD: app
  # ml service roda separado (tcc-back-2)
```

### 6.12 Ordem de Implementação Sugerida

1. Scaffold FastAPI + config + health check
2. Modelos SQLAlchemy + Alembic migrations
3. OAuth Strava (login + callback + sessão cookie)
4. `GET /auth/status` + `POST /auth/disconnect`
5. Strava client + sync de atividades
6. Adapter Strava → schema ML
7. ML client + `GET /api/dashboard`
8. Cache de análise + sync incremental
9. Testes unitários e integração
10. Docker + README + `.env.example`

---

## 7. Assunções e Dependências

| Assunção | Detalhe |
|----------|---------|
| Frontend existente | `run-metrics-mate` será adaptado para consumir esta API |
| ML como serviço separado | `tcc-back-2` exporá FastAPI em porta 8001 |
| Strava app registrada | Usuário criará app em https://www.strava.com/settings/api |
| Histórico mínimo ML | 20 corridas (`MIN_CORRIDAS_PERFIL` no ML) |
| Apenas corridas | Atividades `Run` são consideradas |
| Single region | Deploy simples (TCC); sem multi-região |

---

## 8. Riscos e Mitigações

| Risco | Mitigação |
|-------|-----------|
| ML ainda sem API HTTP | Implementar stub/mock no backend para dev; contrato definido acima |
| Rate limit Strava | Backoff + sync incremental com `after` |
| FC ausente | ML já trata via mediana; backend envia `null` |
| Token expirado mid-request | Refresh automático transparente |
| CORS/cookie em dev | Configurar `SameSite=Lax` e portas fixas (5173/8000) |

---

## 9. Referências

- Strava API: https://developers.strava.com/docs/reference/
- Spec Kit: https://github.com/github/spec-kit
- Serviço ML: repositório `tcc-back-2` (Python, Polars, scikit-learn)
- Frontend: https://github.com/majutestoni/run-metrics-mate
- Dataset de referência: https://www.kaggle.com/datasets/olegoaer/running-races-strava
