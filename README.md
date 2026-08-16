# HistoVision

Система поддержки принятия диагностических решений (СППР) для врача-патологоанатома. Анализирует цифровые гистологические препараты рака молочной железы (H&E-окраска), классифицирует ткань (инвазивная карцинома / доброкачественная), объясняет решение методом GradCAM++ и формирует PDF-заключение.

> **Это инструмент поддержки, не замена врача.** Окончательное заключение всегда формулирует врач-патологоанатом. Метрики модели (AUC-ROC 0.9953, accuracy 97%) получены на датасете BreakHis (одна лаборатория, n=7909) и не являются клинически валидированными.

**On-premise.** Система разворачивается на сервере/ноутбуке медучреждения. Данные пациентов не покидают периметр (152-ФЗ). Инференс работает на CPU — GPU не требуется.

**Область применения v1.0:** только инвазивная карцинома vs доброкачественная ткань, только H&E, только рак молочной железы. DCIS, LCIS, метастазы — вне области.

## Возможности

- Классификация одного патча (H&E-изображение, PNG/JPEG) — результат за секунды.
- Анализ полноформатного WSI-препарата (тайлинг → фильтрация фона → нормализация окраски Macenko → инференс → агрегация) — до ~15 минут на CPU.
- Визуальное объяснение решения (GradCAM++) с топ-N зонами внимания.
- История случаев, подтверждение заключения врачом.
- Генерация PDF-заключения (ReportLab).

## Технологический стек

| Слой | Технология |
|---|---|
| Модель / инференс | PyTorch, timm (EfficientNet-B3), pytorch-grad-cam |
| WSI | OpenSlide (openslide-python) |
| Обработка изображений | OpenCV, Pillow, NumPy |
| Нормализация окраски | собственная реализация метода Macenko |
| Backend / API | FastAPI, SQLAlchemy + SQLite |
| Frontend | React + TypeScript + Vite |
| PDF-отчёты | ReportLab |
| Контейнеризация | Docker + Docker Compose |

Подробности архитектуры и потока данных — [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Алгоритмы (тайлинг, Macenko, GradCAM++, агрегация) с формулами — [docs/ALGORITHMS.md](docs/ALGORITHMS.md). Эндпоинты API — [docs/API.md](docs/API.md). Модель, метрики, обучение — [docs/MODEL.md](docs/MODEL.md).

## Структура проекта

```
histovision/
├── config/               # config.yaml — все пути/пороги/параметры, без хардкода
├── src/
│   ├── wsi/              # чтение и тайлинг WSI (OpenSlide)
│   ├── preprocessing/     # нормализация окраски (Macenko), фильтрация фона (Otsu)
│   ├── inference/         # загрузка модели, классификация патча, WSI-агрегация
│   ├── xai/               # GradCAM++, извлечение топ-N зон
│   ├── report/            # генерация PDF (ReportLab)
│   ├── api/                # FastAPI: роуты, БД (SQLAlchemy), аутентификация
│   └── utils/              # конфигурация, логирование
├── scripts/download_weights.py
├── models/                # веса модели (скачиваются, не в git)
├── data/                  # sqlite + загруженные файлы + PDF (не в git)
├── tests/                 # pytest
├── frontend/               # React + TS + Vite
└── docs/
```

## Требования

- Python 3.11+
- Node.js 20+ (для сборки фронтенда)
- Для полноценного WSI-анализа — нативная библиотека OpenSlide:
  - **Linux (в т.ч. Docker-образ)** — устанавливается через `apt-get install libopenslide0`, уже включено в `Dockerfile.api`.
  - **Windows** — `pip install openslide-python` ставит только Python-обёртку; саму библиотеку нужно скачать отдельно с [openslide.org/download](https://openslide.org/download/) и добавить в `PATH`. Без неё анализ одного патча продолжает работать, WSI-эндпоинт вернёт понятную ошибку (`OpenSlideUnavailableError`) вместо краха — см. `src/wsi/reader.py`.
- Обычный CPU достаточен; GPU не требуется и не используется.

## Установка и запуск (локально, без Docker)

```powershell
# Backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python scripts/download_weights.py      # скачивает models/model.pth (~43 МБ)
uvicorn src.api.main:app --reload       # http://localhost:8000, Swagger: /docs

# Frontend (в отдельном терминале)
cd frontend
npm install
npm run dev                              # http://localhost:5173, проксирует /api на :8000
```

Учётная запись по умолчанию (задаётся в `config/config.yaml`, секции `auth.seed_user`, пароль переопределяется через переменную окружения `HISTOVISION_SEED_USER_PASSWORD`):

- **Логин:** `e.sokolova@clinic.ru`
- **Пароль:** `changeme123` (обязательно сменить перед реальным использованием)

## Запуск через Docker Compose (рекомендуемый способ для on-premise)

```bash
cp .env.example .env    # задать HISTOVISION_JWT_SECRET и HISTOVISION_SEED_USER_PASSWORD
docker compose up --build
```

- Frontend: http://localhost:8080
- API + Swagger: http://localhost:8000/docs

Веса модели (`models/model.pth`) и все пользовательские данные (`data/`) монтируются как volume — не запекаются в образ.

## Конфигурация

Все пути, пороги и параметры модели вынесены в [config/config.yaml](config/config.yaml) — ничего не захардкожено в коде. Два секрета (JWT-ключ и пароль дефолтного пользователя) можно переопределить переменными окружения `HISTOVISION_JWT_SECRET` и `HISTOVISION_SEED_USER_PASSWORD`, не редактируя файл конфигурации.

## Тесты

```powershell
pytest -q
```

Тесты препроцессинга, тайлинга, извлечения GradCAM-зон и API-эндпоинтов не требуют скачанных весов модели — используется нетренированная архитектура (см. `tests/conftest.py`). WSI-пайплайн целиком (`plan_tiles`/`iter_tile_images`) требует установленного OpenSlide и покрывается интеграционно в Docker-окружении.

## Лицензия и данные

Обучающий датасет — BreakHis (n=7909). WSI-пайплайн апробирован на CAMELYON16. Модель и веса — см. [docs/MODEL.md](docs/MODEL.md).
