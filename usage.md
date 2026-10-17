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

## 5. Запуск офлайн-оценки (`ret-eval`)

Прогоняет набор K6 и пишет каталог в `data/eval/runs/` и строку в
`docs/reports/ret_runs.md`.

```bash
poetry run ret-eval \
  --config src/retriever/configs/profiles/dense.yaml \
  --dataset tests/fixtures/k6/k6_sample.jsonl \
  --split dev \
  --variant last
```

| Флаг | По умолчанию | Значение |
|---|---|---|
| `--config` | обязателен | YAML-профиль (`config_id` берётся из имени файла) |
| `--dataset` | обязателен | набор в формате K6 |
| `--split` | — | `train`/`validation` (D6), `dev`/`holdout` (D14) |
| `--variant` | `last` | `last` \| `concat` \| `rewrite` |
| `--output-root` | `data/eval/runs` | куда писать каталоги прогонов |
| `--tune` | выкл. | режим подбора; **запрещён** на validation/test/holdout |
| `--final` | выкл. | разрешает файл test (`mdd_eval_test_v1`) |
| `--workers` | 4 | параллельные запросы |

Ограничения честности (INV-RET-07): строка без `split` — ошибка;
`--tune` на защищённых сплитах — ошибка; файл test без `--final` — ошибка.

## 6. Тестовая коллекция из фикстуры K2 (только для тестов)

Не поставка. Кладёт точки фикстуры в `mdd_exp_test_v1` в **тестовом** Qdrant и
пишет метаданные в sibling-коллекцию (их читает guard). Рабочие алиасы не
трогает.

```bash
poetry run python - <<'PY'
from qdrant_client import QdrantClient
from retriever.tools.load_test_collection import load_fixture, drop_collection

c = QdrantClient(url="http://localhost:6333")
load_fixture(c, collection="mdd_exp_test_v1", embedding_model="BAAI/bge-m3")
# ... тесты/эксперименты ...
drop_collection(c, "mdd_exp_test_v1")   # удалит и *__index_info
PY
```

## 7. Конфиги и профили

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

```bash
poetry run python -c "from retriever.config import load_config; print(load_config('src/retriever/configs/profiles/dense.yaml').config_id)"
# dense@a7cf91ec
```

## 8. Типичные проблемы

| Симптом | Причина / решение |
|---|---|
| `poetry: command not found` | Poetry не установлен — см. https://python-poetry.org/docs/#installation |
| `poetry check` сообщает о рассинхроне с lock | `poetry lock` (или `poetry lock --no-update`, если версии менять не нужно) |
| Integration-тесты пропущены (`skipped`) | Qdrant недоступен. Поднимите его: `docker compose -f docker-compose.test.yml up -d` |
| `ModuleNotFoundError: No module named 'tests'` | Запускайте тесты из корня репозитория; в `pyproject.toml` задан `pythonpath = ["."]` |
| Guard падает «model mismatch» | Коллекция посчитана другой моделью; в ошибке названы обе. Пересоберите коллекцию или поправьте `embedding.model` |
| `/health` отвечает, но `/retrieve` → 503 | Qdrant недоступен или упал реранкер (только при `rerank.enabled: true`) |
| Код 422 | Пустой `query`, `top_k` вне `1..top_k_max`, неизвестное поле фильтра, передано `history` или коллекция не из `allowed_collections` |
| Сервис не стартует без GPU | Это ожидаемо при `embeddings`/`rerank`: стартуют реальные модели. Для разработки без модели используйте тесты с фейковым эмбеддером |

## 9. Быстрый чек-лист

```bash
poetry install --extras dev                               # 1. окружение + dev
poetry run pytest -m "not integration" -q                 # 2. unit-тесты
docker compose -f docker-compose.test.yml up -d           # 3. тестовый Qdrant
poetry run pytest -m integration -q                       # 4. integration
docker compose -f docker-compose.test.yml down -v         # 5. убрать Qdrant
docker compose up --build                                 # 6. сервис целиком
```
