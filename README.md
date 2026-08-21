# HistoVision

Система поддержки принятия диагностических решений (СППР) для врача-патологоанатома. Анализирует цифровые гистологические препараты рака молочной железы (H&E-окраска) и выполняет попиксельную семантическую сегментацию тканей (16 классов BCSS — опухоль, строма, некроз, сосуды и др.), выводит долю площади по каждому классу, вердикт и формирует PDF-заключение.

> **Это инструмент поддержки, не замена врача.** Окончательное заключение всегда формулирует врач-патологоанатом. Сегментационная модель обучена на частичной выборке **BCSS**; итоговый mIoU полноценного обучения ещё не подтверждён — см. [docs/MODEL.md](docs/MODEL.md), раздел «Честность в отношении метрик», прежде чем полагаться на точность предсказаний.
>
> Ветка `master` содержит предыдущую, полноценно обученную версию — бинарный классификатор (EfficientNet-B3, AUC-ROC 0.9953 на BreakHis) с GradCAM++ — как рабочий fallback.

**On-premise.** Система разворачивается на сервере/ноутбуке медучреждения. Данные пациентов не покидают периметр (152-ФЗ). Инференс работает на CPU — GPU не требуется (обучение — требует GPU, см. docs/MODEL.md).

**Область применения:** многоклассовая сегментация тканей молочной железы (16 классов BCSS, см. docs/MODEL.md), только H&E-окраска, только рак молочной железы. Вердикт «злокачественная/доброкачественная» — упрощение поверх карты сегментации (площадь классов `tumor`+`dcis`), не самостоятельное клиническое стадирование.

## Возможности

- Сегментация одного патча (H&E-изображение, PNG/JPEG) — результат за доли секунды.
- Анализ полноформатного WSI-препарата (тайлинг с перекрытием → фильтрация фона → нормализация окраски Macenko → посегментная сегментация → сшивка маски через окно Ханна) — целевой бюджет ~15 минут на CPU.
- Визуализация: маска сегментации, наложенная на изображение препарата, легенда по реально найденным классам, доля площади каждого класса.
- История случаев, подтверждение заключения врачом.
- Генерация PDF-заключения (ReportLab) с маской и разбивкой по классам тканей.

## Технологический стек

| Слой | Технология |
|---|---|
| Модель / инференс | PyTorch, `segmentation-models-pytorch` (DeepLabV3+/ResNet-18) |
| Обучение | `src/training/` — датасет BCSS, взвешенный CE+Dice loss, mIoU/per-class IoU (CLI, для запуска на GPU) |
| WSI | OpenSlide (openslide-python) |
| Обработка изображений | OpenCV, Pillow, NumPy |
| Нормализация окраски | собственная реализация метода Macenko |
| Backend / API | FastAPI, SQLAlchemy + SQLite |
| Frontend | React + TypeScript + Vite |
| PDF-отчёты | ReportLab |
| Контейнеризация | Docker + Docker Compose |

Подробности архитектуры и потока данных — [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Алгоритмы (тайлинг, Macenko, сшивка масок сегментации, агрегация по классам) с формулами — [docs/ALGORITHMS.md](docs/ALGORITHMS.md). Эндпоинты API — [docs/API.md](docs/API.md). Модель, таксономия классов, обучение — [docs/MODEL.md](docs/MODEL.md).

## Структура проекта

```
histovision/
├── config/               # config.yaml, bcss_classes.yaml — пути/пороги/таксономия, без хардкода
├── src/
│   ├── wsi/              # чтение и тайлинг WSI (OpenSlide)
│   ├── preprocessing/     # нормализация окраски (Macenko), фильтрация фона (Otsu)
│   ├── inference/         # сегментация патча/WSI, сшивка маски, вердикт
│   ├── training/          # датасет BCSS, loss, метрики, обучение (офлайн, для GPU-машины)
│   ├── xai/               # GradCAM++ — используется только веткой master (бинарный путь)
│   ├── report/            # генерация PDF (ReportLab)
│   ├── api/                # FastAPI: роуты, БД (SQLAlchemy), аутентификация
│   └── utils/              # конфигурация, логирование
├── notebooks/             # train_segmentation_colab.ipynb — обучение на GPU в Colab
├── models/                # веса модели (не в git — см. docs/MODEL.md)
├── data/                  # sqlite + загруженные файлы + PDF + датасет BCSS (не в git)
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
# Веса сегментационной модели (models/segmentation.pth) не распространяются
# отдельным скриптом — обучаются на стороне пользователя (GPU, см. docs/MODEL.md,
# «Как обучить») и кладутся по пути segmentation_model.weights_path из config.yaml.
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

Веса модели (`models/segmentation.pth`) и все пользовательские данные (`data/`) монтируются как volume — не запекаются в образ.

## Конфигурация

Все пути, пороги и параметры модели вынесены в [config/config.yaml](config/config.yaml) — ничего не захардкожено в коде. Два секрета (JWT-ключ и пароль дефолтного пользователя) можно переопределить переменными окружения `HISTOVISION_JWT_SECRET` и `HISTOVISION_SEED_USER_PASSWORD`, не редактируя файл конфигурации.

## Тесты

```powershell
pytest -q
```

Тесты препроцессинга, тайлинга и API-эндпоинтов не требуют обученных весов модели — используется нетренированная архитектура (см. `tests/conftest.py`). WSI-пайплайн целиком (`plan_tiles`/`iter_tile_images`) требует установленного OpenSlide и покрывается интеграционно в Docker-окружении. `src/training/` покрыт отдельным smoke-тестом (`tests/test_training_smoke.py`) — проверяет, что пайплайн подготовки данных и обучения не падает, не что он даёт хорошую точность.

## Лицензия и данные

Обучающий датасет — BCSS (Breast Cancer Semantic Segmentation, 16-классовая группировка — см. `config/bcss_classes.yaml`). WSI-пайплайн (тайлинг/фильтрация фона/нормализация) апробирован независимо на CAMELYON16, переносится на любой WSI-датасет. Модель, метрики и статус обучения — см. [docs/MODEL.md](docs/MODEL.md).
