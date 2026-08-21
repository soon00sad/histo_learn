# API

FastAPI генерирует интерактивную документацию (Swagger UI) на `/docs` и OpenAPI-схему на `/openapi.json`. Этот документ — текстовое дополнение с описанием потоков использования.

Base URL: `/api/v1`. Все эндпоинты, кроме `/auth/login` и `/health`, требуют заголовок `Authorization: Bearer <token>`.

## Аутентификация

### `POST /auth/login`
Тело: `{ "email": string, "password": string }`
Ответ: `{ access_token, token_type: "bearer", user: { email, full_name, role } }`

### `GET /auth/me`
Данные текущего пользователя по токену.

## Анализ

### `POST /analyze/patch`
Синхронный анализ одного патча (H&E, PNG/JPEG). `multipart/form-data`:

| Поле | Тип | Обязательное |
|---|---|---|
| `file` | файл изображения | да |
| `tissue_type` | строка | нет |
| `ki67` | число 0–100 | нет |
| `er_status`, `pr_status`, `her2_status` | строка | нет |

Ответ (`200`, `AnalysisResult`): `{ case_id, verdict_label, is_malignant, tumor_area_fraction, class_areas: [{name_en, name_ru, color, fraction}], analysis_mode }` — случай уже сохранён в истории. `class_areas` содержит только реально найденные на изображении классы (см. `config/bcss_classes.yaml` для полной таксономии), отсортированные по убыванию доли площади.
`400` — файл не является распознаваемым изображением.

### `POST /analyze/wsi`
Асинхронный анализ полноформатного препарата. Те же поля формы, `file` — WSI-файл (SVS/TIFF/NDPI/…). Отвечает немедленно (`202`) телом `{ "job_id": string }`; сам анализ идёт в фоне (целевой бюджет ~15 минут на CPU — см. docs/MODEL.md за реально измеренным временем на тестовом файле).

## Задачи (WSI)

### `GET /jobs/{job_id}`
Статус фоновой задачи: `{ id, status, stage, progress (0–1), message, case_id, error }`. `status`: `queued | running | done | failed`. `stage`: `tiling | inference | stitching | done` (см. docs/ARCHITECTURE.md, «Поток данных»). Когда `status=done`, `case_id` указывает на созданный случай. Опрашивается фронтендом каждые ~1.5 с на экране «Обработка».

## Случаи

### `GET /cases`
Список случаев (последние сверху), `CaseSummary[]`: `{ id, created_at, tissue_type, verdict_label, is_malignant, tumor_area_fraction, status }`. Query-параметры: `status` (`pending|confirmed`), `verdict` (`malignant|benign`), `search` (подстрока ID).

### `GET /cases/{case_id}`
Полная информация по случаю (`CaseDetail` — `CaseSummary` + ниже): `source_filename, analysis_mode, class_areas [{name_en, name_ru, color, fraction}], ki67, er_status, pr_status, her2_status, report_available`.

### `PATCH /cases/{case_id}`
Тело: `{ "status": "pending" | "confirmed" }` — подтверждение заключения врачом.

### `GET /cases/{case_id}/image`
Исходное изображение препарата (JPEG).

### `GET /cases/{case_id}/mask`
Маска сегментации — индексированный PNG (`mode="P"`), байтовое значение пикселя = индекс класса модели (0–15); непокрытые пиксели прозрачны (`tRNS`-чанк). Не JPEG — сжатие с потерями исказило бы точные ID классов. Фронтенд накладывает её поверх `/image` полупрозрачным слоем (`mix-blend-mode`), а не показывает отдельно.

## Отчёты

### `POST /cases/{case_id}/report`
Явно (пере)генерирует PDF-заключение. Ответ: `{ "report_url": "/api/v1/cases/{id}/report.pdf" }`.

### `GET /cases/{case_id}/report.pdf`
Отдаёт PDF-заключение; если ещё не сформирован — генерирует по запросу.

## Здоровье сервиса

### `GET /health`
`{ "status": "ok", "app": "HistoVision", "version": "..." }`. Без авторизации.
