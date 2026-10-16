"""Write the solution note from actual fitted, replayed and tested artifacts."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "reports/v10"
    selection, metrics, verify, tests, cli, environment = [json.loads((out / name).read_text()) for name in
        ("selection.json", "metrics.json", "verification.json", "tests.json", "cli_verification.json", "environment.json")]
    assert verify["offline_full_inference_byte_identical"] and tests["status"] == cli["status"] == "passed"
    assert metrics["diagnostic_gate_passed"]
    old, new = selection["baseline_v9"], selection["selected"]["metrics"]
    previous, diagnostic = metrics["previous_v9_holdout"], metrics["holdout"]
    info = metrics["submission"]
    packages = "\n".join(f"- {name}=={version}" for name, version in environment["packages"].items())
    text = f'''# Решение v10: собственный ансамбль для определения фишинга

Для загрузки: submissions/submission_content_v10.csv. В нем {info["rows"]} строк, заголовок Id,Predicted, исходный порядок test.csv и бинарные ответы. Положительных ответов: {info["positive_predictions"]}.

Модель: artifacts/v10/ensemble.joblib. Она содержит все обученные компоненты, включая новую SVM; это не таблица публичных ответов. Источники меток модели: только тренировочные данные организатора. В сохраненных HTML-кешах test меток нет. HTML анализируется статически и локально.

## Изменение и результат

Компонент, который изучает слова, фрагменты слов, разметку HTML и URL, заменен на LinearSVC(C=0.5). Числовые признаки берутся из прежнего собственного ансамбля. Для перевода выхода SVM в вероятность применяется sigmoid-калибровка на трех внутренних обучающих folds. Словари и IDF взяты из проверенных собственных C16-моделей для ровно той же обучающей части. Внутри калибровки словарь видит всю внешнюю обучающую часть; внешняя validation при обучении исключена.

Сохранены настройки v9: больший вес URL при короткой странице или заголовке ошибки, признаки страницы, дополнительный текст JS/CSS/ресурсов, правила и статистика только по train. Выбранная доля SVM: {selection["selected"]["text_replacement"]}; порог: {selection["selected"]["threshold"]:.17g}.

| Проверка | v9 F1 | v10 F1 | Ошибки v9 | Ошибки v10 |
| --- | ---: | ---: | ---: | ---: |
| Development OOF, {new["rows"]} строк | {old["f1"]:.9f} | {new["f1"]:.9f} | {old["fp"]+old["fn"]} | {new["fp"]+new["fn"]} |
| Диагностическая часть, {diagnostic["rows"]} строк | {previous["f1"]:.9f} | {diagnostic["f1"]:.9f} | {previous["fp"]+previous["fn"]} | {diagnostic["fp"]+diagnostic["fn"]} |

В OOF исправлено {selection["selected"]["fixed_errors"]} ошибок, появилось {selection["selected"]["new_errors"]}. В диагностической части исправлено {metrics["holdout_fixed_errors"]}, новых {metrics["holdout_new_errors"]}. Development и диагностическая часть уже использовались в прежних экспериментах: эти результаты помогают выбирать модель, но не являются независимой оценкой. Конфигурация и критерий экспорта были заморожены до диагностической проверки.

Относительно v7 изменено {info["prediction_changes_vs_v7"]} ответов, v8 - {info["prediction_changes_vs_v8"]}, v9 - {info["prediction_changes_vs_v9"]}. Турнирная оценка v10 пока не получена. Подтвержденная пользователем публичная оценка v7/v8: 0.995080. Одинаковая оценка разных файлов сама по себе не показывает, какие ошибки изменились на публичной и приватной частях.

## Воспроизведение ответов

Python 3.12.10, проверенная платформа: {environment["platform"]}. Распакуйте архив, откройте каталог phishing-hackathon и создайте отдельное окружение:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m phishing_url.v10_cli \\
  --model artifacts/v10/ensemble.joblib \\
  --test data/test.csv --sample data/sample.csv \\
  --content-cache data/content_test.joblib \\
  --extra-cache data/extra_html_test.joblib \\
  --url-cache data/content_url_test.joblib \\
  --output reproduced_v10.csv
```

Рабочее окружение проверено по установленным версиям. Установка зависимостей в новое окружение отдельно не выполнялась. Проверка архива использует текущее окружение с заново распакованными исходниками, моделями и данными.

SHA256 ожидаемого CSV: {info["sha256"]}. SHA256 полной модели: {verify["model_sha256"]}.

## Обучение и проверенные границы

Исходный подбор: experiments/v10_margin/run_oof.py, запуск один раз, три внешних folds, всего 12 SVM fits с учетом внутренней калибровки. Только 50977 development строк; holdout и 178 добавленных train строк исключены. Отбор трех заранее объявленных долей:

```sh
.venv/bin/python -m experiments.v10_margin.select
```

Прямой запуск файла select.py конфликтует с именем стандартного Python-модуля select; используйте форму с -m. Неуспешный прямой запуск сохранен в selection_initial_run.log, обучений из-за него не добавилось. Скрипты подбора защищены от случайного повторного запуска и перезаписи результатов.

scripts/train_v10.py обучил диагностическую SVM на 50977 строках и финальную на всех 63900 тренировочных строках, включая 178 добавленных организатором. Это еще 8 SVM fits с калибровкой, не переобучение всех старых компонентов. Для отдельного повторного обучения новой SVM в другой файл:

```sh
.venv/bin/python scripts/retrain_v10_margin.py --stage full --output artifacts/retrained_margin.joblib
```

В этом скрипте используются архивированные C16-словари с проверкой точного состава train; старый файл не перезаписывается. Новый файл содержит дополнительный SVM-компонент, а не весь ансамбль. Источники обучения прежних компонентов находятся в scripts/train_v7.py и более ранних скриптах, их модели и настройки включены. Численное совпадение повторного обучения на другой платформе не гарантируется. Повторное обучение через вспомогательный скрипт в этом раунде дополнительно не запускалось; его интерфейс проверен. Реальное обучение выполнено через train_v10.py с тем же классом и параметрами.

Также с помощью Астры проверен один NB-вариант на inner-train: критерий улучшения не выполнен. Он не входит в итоговый ансамбль. Протокол, исходники, метрики и проверки: experiments/v10_nb/REPORT.md. Исходный отрицательный inner-результат SVM тоже сохранен: новая версия проверяет его взаимодействие с полным ансамблем, не переименовывая прежний отрицательный результат в успех.

Проверено: {tests["tests"]} тестов, повторный расчет всей development-смеси, scope и словари обучающих моделей, вся диагностическая выборка, побайтовое совпадение повторного CSV для всех {info["rows"]} test строк. Реальная CLI-команда проверена на 8 test строках при заблокированной сети. Отдельная проверка архива записывается в phishing-hackathon-solution-v10.zip.verified.json рядом с архивом.

Основные версии из проверенного окружения:

{packages}

Отчеты: reports/v10/selection.json, metrics.json, verification.json, environment.json, tests.json, cli_verification.json. Внешние опубликованные метки, submission_reference.csv и публичные SQL-ответы не использованы и не включены в архив.
'''
    assert not any(char in text for char in "—–«»„“”…")
    (ROOT / "README_V10.md").write_text(text)
    print(json.dumps({"readme": str(ROOT / "README_V10.md"), "tests": tests["tests"],
                      "submission_sha256": info["sha256"]}))


if __name__ == "__main__":
    main()
