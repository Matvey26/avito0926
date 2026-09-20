# Mean-minmax retrieval fusion

Итоговое решение объединяет три локальных retriever-канала, выбранных в
экспериментах 0008, 0009, 0011 и 0012. Оно не использует benchmark-разметку,
не обращается к внешним API во время расчёта и детерминированно создаёт
`answer.csv`.

## Подход

Каждый retriever ищет только внутри категории запроса и возвращает 500
кандидатов с исходными оценками:

1. **Word BM25** — русская лемматизация через `pymorphy3`, униграммы,
   заголовок + параметры + описание объявления; запрос + параметры поиска.
2. **compress-fastText** — 300-мерные GeoWAC-векторы из открытого проекта
   [`avidale/compress-fasttext`](https://github.com/avidale/compress-fasttext).
   Заголовок и запрос кодируются L2-нормализованной суммой word-векторов,
   затем считается точное cosine-сходство.
3. **Char BM25** — `char_wb` 4-граммы, заголовок + параметры + описание;
   запрос + параметры поиска.

Для каждого запроса оценки каждого канала независимо min-max нормализуются.
Кандидат, отсутствующий в канале, получает ноль; три значения усредняются.
После fusion к оценке объявления из точной локации запроса прибавляется 0.5,
затем выбираются 50 лучших кандидатов. Категория запроса 0 отображается в 114
по наблюдению из train.

На фиксированной локальной cold/warm-валидации эксперимента 0012 стратегия
получила Recall@50 0.65079 / 0.65095 и смешанную оценку 0.65085. Это локальный
ориентир, а не результат тестирующей системы.

## Воспроизведение

Из корня проекта:

```bash
uv venv .venv
uv pip install --python .venv/bin/python -r solution/requirements.txt
.venv/bin/python solution/download_model.py
.venv/bin/python solution/run.py
.venv/bin/python solution/validate.py
```

`download_model.py` нужен только один раз: он скачивает открытый файл модели и
проверяет его SHA-256. Основной pipeline работает полностью локально, читает
`assets/dataset/benchmark_queries.parquet` и
`assets/dataset/benchmark_items.parquet`, а затем записывает `answer.csv` в
корень проекта.

Для нестандартных путей доступны параметры:

```bash
.venv/bin/python solution/run.py \
  --data-dir assets/dataset \
  --model-path assets/models/compress_fasttext/geowac_tokens_sg_300_5_2020-100K-20K-100.bin \
  --output answer.csv
```

Проверка требует ровно две колонки, полный набор уникальных `query_id`, от 1
до 50 неповторяющихся корректных `item_id` на строку и принадлежность каждого
идентификатора benchmark-корпусу.

## Использованные open-source компоненты

- Okapi BM25 и mean-minmax fusion реализованы локально;
- scikit-learn используется для построения разреженных счётчиков;
- `pymorphy3` — для русской лемматизации;
- `compress-fasttext` и GeoWAC — для dense word-векторов;
- NumPy, SciPy, pandas и PyArrow — для вычислений и чтения данных.
