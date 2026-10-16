"""Explain the actually verified v6 model in Russian, without assumed public gain."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    metrics = json.loads((ROOT / "reports/v6/metrics.json").read_text())
    selection = json.loads((ROOT / "reports/v6/selection.json").read_text())
    verified = json.loads((ROOT / "reports/v6/verification.json").read_text())
    tests = json.loads((ROOT / "reports/v6/tests.json").read_text())
    previous = json.loads((ROOT / "reports/v5/metrics.json").read_text())
    assert verified["offline_full_inference_byte_identical"] and tests["status"] == "passed"
    chosen = selection["selected"]
    assert chosen["config"]["base_version"] == 5 and chosen["config"]["extra_text_weight"] == .1
    assert chosen["config"]["extra_boost_weight"] == 0 and not chosen["config"].get("neighbor_adjustment")
    err = lambda m: m["fp"] + m["fn"]
    content = f'''# Кандидат v6: дополнительная модель по JavaScript, CSS и метаданным

Public F1 v3 и v5, сообщенный участником: 0.994425. V4 получил 0.994096. Public и private F1 v6 пока неизвестны. Это новый проверенный кандидат для загрузки, его преимущество в турнире еще не подтверждено. Архивы прежних версий сохранены отдельно.

## Что изменилось и что получилось

V6 сохраняет шесть обученных компонентов v5 и добавляет седьмой классификатор. Он читает статический текст скриптов, CSS, метатегов и адресов подключаемых ресурсов. Прежняя модель использовала текст страницы, URL, разметку и числовые признаки, но большую часть содержимого script/style пропускала. Страницы из архива рассматриваются как строки; JavaScript не исполняется, ссылки и DNS не запрашиваются.

| Проверка | F1 v5 | F1 v6 | Ошибки v5 | Ошибки v6 |
|---|---:|---:|---:|---:|
| Development OOF, 50977 URL | {selection['baseline_v5']['f1']:.9f} | {chosen['metrics']['f1']:.9f} | {err(selection['baseline_v5'])} | {err(chosen['metrics'])} |
| Прежняя контрольная выборка, 12745 URL | {previous['holdout']['f1']:.9f} | {metrics['holdout']['f1']:.9f} | {err(previous['holdout'])} | {err(metrics['holdout'])} |
| Контрольные страницы без точного совпадения с train, 9018 URL | {previous['unseen_content_metrics']['f1']:.9f} | {metrics['unseen_content_metrics']['f1']:.9f} | {err(previous['unseen_content_metrics'])} | {err(metrics['unseen_content_metrics'])} |

На OOF исправлено {chosen['fixed_v5_errors']} ошибок v5 и добавлено {chosen['new_errors']} новых. Улучшение есть на двух фолдах, на одном результат хуже. На контрольной выборке исправлено {metrics['holdout_fixed_errors']} ошибок и добавлено {metrics['holdout_new_errors']}, общее число не изменилось. FP уменьшилось с {previous['holdout']['fp']} до {metrics['holdout']['fp']}, FN выросло с {previous['holdout']['fn']} до {metrics['holdout']['fn']}. Поэтому контрольный F1 v6 немного ниже v5. Сильного подтверждения преимущества нет.

CSV содержит {metrics['submission']['rows']} ответов в исходном порядке test, {metrics['submission']['positive_predictions']} из них равны 1. От v5 отличаются {metrics['submission']['prediction_changes_vs_v5']} ответов. Порог класса 1: {chosen['threshold']:.12f}.

## Как работает новая часть

Из первого миллиона символов каждой страницы извлекаются четыре текстовых поля. Скрипты и CSS ограничиваются 18000 символами каждый, метаданные 6000, адреса ресурсов 16000. Длинные последовательности из 60 и более разрешенных символов, включая base64, заменяются маркером LONGTOKEN. При превышении лимита сохраняются начало и конец текста. Некорректные адреса ресурсов получают признак INVALID и не останавливают разбор.

TF-IDF превращает слова и пары соседних слов в числа. Частый на многих страницах признак получает меньший вес, редкий и характерный для страницы больший. Словари и IDF строятся только на обучающих строках текущего разбиения. Для каждого поля оставляется не более 150000 признаков; слово должно встретиться хотя бы в двух обучающих документах. В отличие от старого словаря по подвыборке, новый классификатор использует все обучающие строки для построения словарей.

Логистическая регрессия учится давать положительный вес признакам, связанным с меткой 1, и отрицательный признакам класса 0. Ее настройки: C=4, solver=liblinear, max_iter=500. C управляет силой регуляризации: более низкое значение сильнее ограничивает веса и помогает сдерживать подгонку под редкие признаки.

В отдельности новая модель слабее v5: ее OOF F1 = 0.957085881. Но ошибки различаются. Поэтому она получает только 10% веса:

```text
p_v6 = 0.9 * p_v5 + 0.1 * p_extra_html
Predicted = 1, если p_v6 >= 0.467248472469, иначе 0
```

В этой формуле p_v5 уже содержит прежнюю смесь, обучающие правила и контекстные оценки. Если прежнее правило или точное совпадение URL с train дало 0 или 1, этот ответ сохраняется. Правила не ослаблялись. Порог выбран на OOF, а не на метках контрольной выборки или test. Оценка модели не является гарантированной вероятностью реальной атаки для отдельной ссылки.

## Как читать метрики

TP - правильно найденный фишинг. FP - безопасный URL, ошибочно названный фишингом. FN - пропущенный фишинг. Precision показывает, какая доля обнаружений верна; recall - какая доля фишинга найдена. F1 объединяет precision и recall: `2 * TP / (2 * TP + FP + FN)`. Это не доля правильных ответов и не вероятность того, что каждая ссылка определена верно.

OOF означает прогнозы на данных, которых соответствующая модель не видела при обучении. Development делится на три части, каждая прогнозируется моделью, обученной на остальных двух. Веса и порог подбираются по этим прогнозам. Поскольку на OOF сравниваются несколько вариантов, итоговая оценка может быть оптимистичной. Контрольная выборка тоже использовалась в прежних экспериментах, поэтому это сравнительная диагностика, а не новый независимый тест. Отдельное новое обучение с непересекающимися доменами для v6 не проводилось.

## Что было проверено и отклонено

Близкие страницы по TF-IDF проверены в четырех представлениях: текст, разметка, их смесь и URL. Внутреннюю базовую модель они не улучшили. Полная OOF проверка близких URL исправила лишь одну ошибку v5; в выбранной смеси этот метод не помог. CatBoost с дополнительными числовыми признаками HTML ухудшал внутреннюю смесь при всех проверенных ненулевых весах. Ограничения новой текстовой модели по уверенности и длине видимого текста также не превзошли обычную смесь. Все результаты сохранены, эти методы в финальный прогноз не входят.

Новые HTML поля извлечены для 63900 обучающих и 16000 тестовых строк. Полное исходное содержимое каждого выбранного URL сверено по SHA256 с прежним кэшем. Метки берутся только из переданного train, дополнительные 178 обучающих URL добавляются только при финальном обучении. Меток test и внешнего опубликованного справочника в модели нет. Новых вызовов Astra или субагентов: 0.

## Воспроизведение готового ответа

Архив содержит компактные данные, исходники, модели, конфигурацию и результаты. Большой исходный HTML CSV организатора для повторного прогноза не нужен. Команды выполняются из распакованной папки phishing-hackathon. Окружение фактического запуска записано в reports/v6/environment.json.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup_openmp.py
.venv/bin/python -m phishing_url.v6_cli \\
  --model artifacts/v6/ensemble.joblib \\
  --cache data/content_test.joblib \\
  --extra-cache data/extra_html_test.joblib \\
  --url-features data/content_url_test.joblib \\
  --sample data/sample.csv \\
  --output submissions/reproduced_content_v6.csv
.venv/bin/python scripts/verify_v6.py
.venv/bin/python -m pytest -q
```

На текущем окружении прошли {tests['tests']} тестов. Сохраненные модели повторно проверены на всех 12745 контрольных строках и всех 16000 тестовых строках при запрещенных сетевых соединениях. Весь итоговый CSV совпал побайтно. Для новых OOF моделей проверены границы обучения по контрольным суммам и заново вычислены первые 32 прогноза каждого фолда. Внешние метки test не использовались ни при обучении, ни при этих проверках.

## Повторное обучение

```bash
.venv/bin/python scripts/train_v6.py --refit-new-models
.venv/bin/python scripts/verify_v6.py
```

Команда заново обучает добавленную модель на development и на всем train. Базовые модели v5 включены в архив; их обучение описано в README_V5.md и README_CONTENT.md. При повторном обучении на другой платформе побайтовая идентичность не гарантируется.

Для повторения поиска на оригинальных development разбиениях:

```bash
.venv/bin/python experiments/v6/screen_neighbors.py
.venv/bin/python experiments/v6/screen_extra.py --kind text
.venv/bin/python experiments/v6/screen_extra.py --kind boost
.venv/bin/python experiments/v6/oof_extra.py --kind text
.venv/bin/python experiments/v6/oof_url_neighbors.py
.venv/bin/python experiments/v6/select_extra.py
.venv/bin/python experiments/v6/gated_search.py
.venv/bin/python scripts/train_v6.py --refit-new-models
```

Воспроизведение использует переданные компактные кэши с фиксированными контрольными суммами. Для повторного извлечения новых полей из исходного архива есть scripts/extract_extra_html.py с параметром --archive. Сам сырой архив в ZIP не включен. Отдельный scripts/verify_package_v6.py проверяет CRC, SHA256 каждого файла, импорт распакованных исходников, тесты и автономный прогноз восьми реальных строк.

Подробные числа: reports/v6/selection.json, metrics.json, verification.json, tests.json. История турнирных результатов: reports/v5/score_feedback.json и tournament_feedback.json. Справочный ответ с public F1 1.0 по опубликованным меткам относится к отдельному способу, который не использовался в обученной модели v6.
'''
    (ROOT / "README_V6.md").write_text(content, encoding="utf-8")
    (ROOT / "reports/v6/RUN_STATUS.md").write_text(
        f"Проверен новый кандидат v6. OOF F1 {chosen['metrics']['f1']:.9f}, holdout F1 {metrics['holdout']['f1']:.9f}.\n"
        "Public/private F1 неизвестны. У v3 и v5 public F1 0.994425 по сообщениям участника.\n"
        "Кандидат меняет 13 ответов v5. Новых вызовов Astra: 0.\n", encoding="utf-8")
    print("README_V6.md generated from verified actual results.")


if __name__ == "__main__":
    main()
