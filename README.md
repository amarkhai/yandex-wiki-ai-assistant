# AI-ассистент для документации из wiki Яндекса с использованием RAG и LangChain

## Руководство по запуску ассистента

### Шаг 1 - выгрузка информации из Wiki

#### 1. Установка Gramax CLI

Gramax CLI можно использовать двумя способами: без установки через npx или с глобальной установкой.
Без установки. Для использования выполните: ```npx gramax-cli <command> [options]```
Глобальная установка. Для регулярного использования выполните: ```npm install -g gramax-cli.```

После этого команды CLI доступны из любой папки.

#### 2. Получение данных из Yandex Wiki
 - Откройте Yandex Wiki и войдите в систему.
 - Нажмите F12, чтобы открыть консоль разработчика.
 - Перейдите на вкладку Network → Fetch/XHR.
 - В левой панели выберите любую статью. В консоли появится запрос getPageDetails.
 - Щелкните правой кнопкой мыши на запросе → Копировать → Копировать как fetch (Node.js).
 - Сохраните скопированный код в текстовом файле.

#### 3. Настройка Gramax CLI
Скопируйте файл gramax.config.yaml.dist в gramax.config.yaml
Заполните шаблон данными, которые скопировали из предыдущего этапа

```yaml
import:
   yandex:
      headers:
            "x-csrf-token": "e8399826638e73245k0f1cfe5a944b87683bbs10:1754349881"
            "x-org-id": "1331068"
            cookie: "yandexuid=8626904001695300517; yuidss=8626904001695300517; gdpr=0; _ym_uid=1695303700525969138; yandex_login=name@name.ru;...924b99683bbb10%33878179541"
            "x-collab-org-id": "cf3c7652-ae27-455f-a126-9de4f12cegaa"
```

Параметр x-collab-org-id не обязателен. Если у вас этого значения нет, просто не указывайте данный параметр.

#### 4. Запуск
Откройте консоль и выполните команду:

```shell
npx gramax-cli import yandex-wiki --destination ./var --config . --raw
```
Где:

--destination, -d — путь до папки, в которую экспортируются статьи из Yandex Wiki.

--config, -c — путь до папки, в которой лежит файл gramax.config.yaml.

--raw, -r — параметр, который выключает трансформацию Markdown под формат Gramax. Если указан — статьи экспортируются в разметке Yandex Wiki. Если не указан — в разметке Gramax.

### Шаг 2 - Заполняем векторную БД

#### 1. Устанавливаем зависимости

   ```bash
   python -m venv .venv && source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
   pip install -r requirements.txt
   ```

#### 2. Создаем и заполняем БД

   ```bash
   python ingest_md.py --docs_dir var/out --out_dir vector_store
   ```

   Common options:

   * `--glob "**/*.md"` to control file discovery.
   * `--chunk_size 900 --chunk_overlap 120` to tune chunking.
   * `--embed_model sentence-transformers/all-MiniLM-L6-v2` to switch models.

### Шаг 3 - Запускаем агент

#### 1. Добавляем OpenAI API KEY

   *  `cp .env.example .env` и меняем `OPENAI_API_KEY`.
   * Можно опционально поменять модель `OPENAI_MODEL` (по умолчанию `gpt-4o-mini`).

#### 2. Запуск агента

   ```bash
   python app.py
   ```
---