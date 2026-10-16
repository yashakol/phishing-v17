"""Write the explanation from actual v11 training and verification reports."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    out = ROOT / "reports/v11"
    selection, metrics, verify, tests, cli, environment = [json.loads((out / name).read_text()) for name in
        ("selection.json", "metrics.json", "verification.json", "tests.json", "cli_verification.json", "environment.json")]
    assert verify["offline_full_inference_byte_identical"] and tests["status"] == cli["status"] == "passed"
    assert metrics["diagnostic_gate_passed"] and selection["original_gate_passed"] is False
    old_dev = json.loads((ROOT / "reports/v10/selection.json").read_text())["selected"]["metrics"]
    new_dev = selection["selected"]["metrics"]
    old_hold, new_hold = metrics["previous_v10_holdout"], metrics["holdout"]
    chosen, info = selection["selected"], metrics["submission"]
    text = f'''# Собственное решение v11 для обнаружения фишинга

Файл для отправки: submissions/submission_content_v11.csv. В нем {info["rows"]} строк с заголовком Id,Predicted и исходным порядком test.csv. Положительных ответов: {info["positive_predictions"]}. По сравнению с v10 изменено {info["prediction_changes_vs_v10"]} ответов. Файл отправляет участник.

Сама модель: artifacts/v11/ensemble.joblib. Это сохраненный ансамбль обученных классификаторов, словарей и правил только по разрешенному train, а не таблица найденных публичных ответов. В test-кешах метки отсутствуют. Страницы анализируются локально, их скрипты не выполняются и сайты не посещаются.

## Что изменилось

Семь собственных обученных компонентов дают вероятности фишинга по видимому тексту, фрагментам слов и HTML, URL, числовым признакам страницы и дополнительным признакам JS/CSS/ресурсов. Раньше их веса были заданы вручную. Новая небольшая логистическая регрессия учится учитывать согласие компонентов, вид страницы и статистику доменов/заголовков, рассчитанную только на тренировочной части. Вход новой модели содержит 35 числовых признаков. Ее C=0.1 ограничивает свободу подгонки коэффициентов.

Мета-модель обучалась на {metrics["meta_training_rows"]} development строках с out-of-fold предсказаниями: базовая модель для каждого такого предсказания не обучалась на самой этой строке. При проверке мета-модели использованы три внешних folds, но схема не полностью вложенная: базовые модели для других обучающих строк могут видеть строки проверяемого meta-fold. Поэтому development F1 нельзя считать независимой оценкой качества.

Новая вероятность участвует с весом {chosen["weight"]}. Старый необработанный ансамбль участвует с весом {1-chosen["weight"]}. Затем применяются прежние правила, основанные на train, мягкая статистика контекста, дополнительный HTML-текст и иерархия URL. Правила не изучают метки test. Порог, выбранный только на development: {chosen["threshold"]:.17g}.

В финальном файле базовые модели ранее были обучены на 63900 разрешенных тренировочных строках. Новая мета-модель по-прежнему использует только development OOF, включая 50977 строк, без 12745 диагностических и 178 добавленных примеров. Эти размеры относятся к разным уровням ансамбля.

## Измеренный результат и ограничения

| Проверка | v10 F1 | v11 F1 | Ошибки v10 | Ошибки v11 |
| --- | ---: | ---: | ---: | ---: |
| Development OOF, {new_dev["rows"]} строк | {old_dev["f1"]:.9f} | {new_dev["f1"]:.9f} | {old_dev["fp"]+old_dev["fn"]} | {new_dev["fp"]+new_dev["fn"]} |
| Повторно используемая диагностика, {new_hold["rows"]} строк | {old_hold["f1"]:.9f} | {new_hold["f1"]:.9f} | {old_hold["fp"]+old_hold["fn"]} | {new_hold["fp"]+new_hold["fn"]} |

В development исправлено {chosen["fixed_errors"]} ошибок, появилось {chosen["new_errors"]}. Ошибки по folds: 144/146/140 у v10, 144/139/136 у нового варианта. При прежнем фиксированном пороге новый вариант дает 424 ошибки против 430. В диагностике исправлено {metrics["holdout_fixed_errors"]}, появилось {metrics["holdout_new_errors"]} ошибок.

Первоначальный строгий отбор двух моделей и трех весов не пройден: требовалось минимум 8 исправленных ошибок и при новом, и при прежнем пороге. Этот отрицательный результат сохранен в experiments/v11_meta/selection.json. Для одного отдельного исследовательского продолжения выбран более простой линейный вариант с весом 0.25: он улучшил две части и не ухудшил третью, в отличие от более сложной лучшей по настроенному F1 модели. До загрузки диагностической модели зафиксированы единственная конфигурация, порог и условие выпуска: F1 выше v10 и минимум на 3 диагностических ошибки меньше. После диагностики параметры не менялись.

Обе локальные части уже использовались в предыдущем поиске. Это диагностический сигнал для следующей попытки, а не доказательство будущего турнирного улучшения. Пользователь сообщил повторную публичную оценку 0.995080 для предыдущего решения; оценка v11 неизвестна. Метки test и приватная турнирная таблица недоступны этому обучению. Новая версия может улучшить или ухудшить турнирный результат.

## Как получить тот же CSV

Проверенная платформа: {environment["platform"]}. Версия Python в рабочем окружении: {environment["python"].split()[0]}. Откройте каталог phishing-hackathon из архива:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m phishing_url.v11_cli \\
  --model artifacts/v11/ensemble.joblib \\
  --test data/test.csv --sample data/sample.csv \\
  --content-cache data/content_test.joblib \\
  --extra-cache data/extra_html_test.joblib \\
  --url-cache data/content_url_test.joblib \\
  --output reproduced_v11.csv
```

SHA256 ожидаемого CSV: {info["sha256"]}. SHA256 модели: {verify["model_sha256"]}.

Все {environment["requirement_pin_count"]} версии зависимостей в requirements.txt совпадают с установленными. Новая установка зависимостей отдельно не выполнялась. Проверка переносимости использует текущее окружение с заново распакованными исходниками, моделями и данными.

Для повторного обучения именно нового компонента в отдельный файл:

```sh
.venv/bin/python scripts/retrain_v11_meta.py --output artifacts/retrained_v11_meta.joblib
```

Скрипт использует сохраненную таблицу собственных OOF-признаков с проверкой хэшей и обучает один новый классификатор. Он сохраняет отдельный компонент, а не весь ансамбль. Полный путь первоначального обучения: experiments/v11_meta/search.py и scripts/train_v11.py. Скрипты прежних компонентов, исходные параметры, модели и результаты включены в архив. Защищенные первоначальные скрипты не перезаписывают уже завершенный эксперимент.

## Проверки и помощь Астры

Пройдено {tests["tests"]} тестов. Повторно рассчитаны все development meta-folds, их обучающие области и итоговая смесь. Повторено предсказание сохраненной диагностической модели. Для всех 16000 test строк получен побайтово тот же CSV. Настоящая CLI-команда также проверена на 8 реальных test строках с блокировкой сети. Проверка распакованного архива записывается рядом с архивом в phishing-hackathon-solution-v11.zip.verified.json.

Астра выполнила диагностический аудит доступных признаков train и unlabeled test без обучения моделей или поиска порога. Он дважды дал совпадающие результаты и хэши: experiments/v11_shift/REPORT.md и verification.json. В development 420 из 430 ошибок прежней версии приходятся на выход обученных моделей вне жестких правил. Эта проверка помогла выбрать направление поиска; она не знает ошибок test и не оценивает турнирный F1.

Первый запуск интеграции остановился до обучения и оценки: строгий загрузчик отказался читать имена столбцов как объектный массив NPZ. Схема теперь берется из того же проверенного JSON; числовые массивы по-прежнему загружаются с allow_pickle=False. Настройки модели и порог не менялись. Подробности: reports/v11/integration_repair.json. Это технический ремонт загрузки, не дополнительный поиск кандидата.

Основные отчеты: reports/v11/selection.json, integration_protocol.json, meta_fit.json, metrics.json, verification.json, environment.json, tests.json и cli_verification.json. Опубликованные внешние метки, публичный SQL и submission_reference.csv не использованы и не входят в архив.
'''
    assert not any(char in text for char in "—–«»„“”…")
    (ROOT / "README_V11.md").write_text(text)
    print(json.dumps({"readme": str(ROOT / "README_V11.md"), "tests": tests["tests"],
                      "submission_sha256": info["sha256"]}))


if __name__ == "__main__":
    main()
