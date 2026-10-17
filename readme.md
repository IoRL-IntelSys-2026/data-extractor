# Как запускать проект и тесты (Poetry)

Практическая инструкция по ретриверу.

## 1. Требования

- Python ≥ 3.11 (проверено на 3.14).
- **Poetry ≥ 2.0** — https://python-poetry.org/docs/#installation
- Docker — для Qdrant (D1) и для integration-тестов.
- GPU (опционально) — только если реально считаете эмбеддинги/реранкер моделью.
  Unit-тесты и CI модель **не скачивают**: вместо неё используется фейковый
  эмбеддер на готовых векторах из фикстуры K2.

## 2. Установка окружения

Из корня репозитория:

```bash
# 2.1. Создать виртуальное окружение внутри проекта и поставить базовые
#      зависимости (qdrant-client, fastapi, pydantic, pyyaml).
poetry install

# 2.2. Для разработки и тестов сразу поставьте extras "dev"
#      (pytest, pytest-cov) — они нужны для запуска тестов.
poetry install --extras dev
```

`poetry install` ставит пакет в editable-режиме из `src/`, поэтому обе части —
`retriever` и `rag_common` — становятся importable, и появляются консольные
команды:

| Команда | Что делает |
|---|---|
| `retriever-api` | запускает FastAPI-сервис `/retrieve`, `/health` |
| `ret-eval` | офлайн-прогон оценки поиска (раннер) |

Все команды ниже запускаются через `poetry run ...` (или внутри `poetry shell`).

### Дополнительные (тяжёлые) группы

Они объявлены как extras, поэтому ставятся только при необходимости:

```bash
poetry install --extras embeddings   # FlagEmbedding + torch — реальный bge-m3
poetry install --extras rerank       # FlagEmbedding + torch — реранкер
poetry install --extras eval         # pandas — для отчётов
poetry install --all-extras          # всё сразу (dev/embeddings/rerank/eval)
```

Зависимости и их версии зафиксированы в `poetry.lock`; диагностика конфига:

```bash
poetry check        # проверяет pyproject.toml и согласованность с poetry.lock
poetry show         # список установленных пакетов
poetry lock         # перегенерировать lock, если конфиг изменился
```

## 3. Запуск сервиса

Сервис при старте (UC-RET-02) читает конфиг, грузит модель, проверяет коллекции
через guard и делает прогревочный запрос. Если коллекция посчитана другой
моделью — процесс падает с понятной ошибкой (INV-RET-03).

### Вариант А. Всё через docker-compose (Qdrant + сервис)

```bash
docker compose up --build
```

- Сервис: http://localhost:8001 (`RETRIEVER_URL` для команды 4)
- Qdrant: http://localhost:6333
- Swagger UI сервиса: http://localhost:8001/docs

### Вариант Б. Qdrant отдельно, сервис локально (через Poetry)

```bash
# 1. поднять только Qdrant
docker compose up -d qdrant

# 2. запустить сервис в окружении Poetry
poetry run retriever-api
# или: poetry run python -m retriever.api
```

### Переменные окружения сервиса

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `RETRIEVER_CONFIG` | `src/retriever/configs/retriever.yaml` | путь к YAML-профилю |
| `QDRANT_URL` | `http://localhost:6333` | адрес Qdrant |
| `QDRANT_API_KEY` | не задан | ключ Qdrant, если включена аутентификация |
| `RETRIEVER_PORT` | `8001` | порт сервиса |
| `LOG_LEVEL` | `INFO` | уровень логов |

### Проверка вручную

```bash
curl http://localhost:8001/health
curl -X POST http://localhost:8001/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "how to apply for a dorm", "top_k": 5}'
```

В каждом ответе есть `config_id` и `latency_ms`; при `debug: true` — поле
`debug` с временами этапов и кандидатами до реранкера.

## 4. Запуск тестов

Тесты вызываются через `poetry run pytest`. Если включён `poetry shell`, можно
опускать `poetry run`.

### Всё сразу

```bash
poetry run pytest
```

По умолчанию прогоняются все тесты. Integration-тесты **автоматически
пропускаются** (`pytest.skip`), если Qdrant на `http://localhost:6333`
недоступен, — поэтому обычный прогон зелёный и без сервера.

### Только unit-тесты (без Qdrant)

```bash
poetry run pytest -m "not integration"
```

Это тесты конфига и `config_id` (T-RET-02, 04), поиска и фильтров на мок-клиенте
(T-RET-05), постобработки и `no_relevant` (T-RET-08, 10, 11), запрета LLM-импортов
(T-RET-06), честных сплитов и `latency_stats` (T-RET-07, 17), вариантов запроса
(T-RET-19), схем K3 и HTTP-сервиса (T-RET-01).

### Integration-тесты (нужен живой Qdrant)

Поднимите изолированный Qdrant для тестов (без volumes, чисто):

```bash
docker compose -f docker-compose.test.yml up -d
```

Затем:

```bash
# адрес по умолчанию — http://localhost:6333
poetry run pytest -m integration -v

# или явный адрес тестового Qdrant
QDRANT_TEST_URL=http://localhost:16333 poetry run pytest -m integration -v
```

Фикстура сама загружает тестовую коллекцию `mdd_exp_test_v1` и удаляет её после
прогона (включая sibling-коллекцию `mdd_exp_test_v1__index_info`). Integration
покрывают guard (T-RET-03), фильтры в реальных запросах (T-RET-05),
воспроизводимость прогона (T-RET-09) и попадание известного чанка в top-k
(T-RET-01).

Убрать тестовый Qdrant:

```bash
docker compose -f docker-compose.test.yml down -v
```

### Один тест / один файл

```bash
poetry run pytest tests/test_config.py -v
poetry run pytest tests/test_integration_qdrant.py::test_guard_ok -v
```

### Покрытие (для отчёта)

```bash
poetry install --extras dev
poetry run pytest --cov=retriever --cov=rag_common --cov-report=term-missing
```

### Маркер `integration`

Зарегистрирован в `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = ["integration: tests that need a running Qdrant"]
testpaths = ["tests"]
pythonpath = ["."]   # чтобы работал импорт "from tests.conftest import ..."
```


## 5. Конфиги и профили

| Файл | Профиль |
|---|---|
| `src/retriever/configs/retriever.yaml` | рабочий конфиг сервиса (алиасы) |
| `src/retriever/configs/profiles/dense.yaml` | baseline: только dense |
| `src/retriever/configs/profiles/sparse.yaml` | только sparse |
| `src/retriever/configs/profiles/hybrid_rrf.yaml` | гибрид, RRF `k=60` |
| `src/retriever/configs/profiles/hybrid_rrf_rerank.yaml` | гибрид + реранкер |
| `src/retriever/configs/profiles/rudn.yaml` | профиль РУДН (гибрид, фильтры) |

`config_id` руками не пишется: это `<имя профиля>@<первые 8 символов хеша>`.
Проверить его для профиля:
