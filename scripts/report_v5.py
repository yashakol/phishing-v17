"""Write the final Russian explanation from verified results, without guessed scores."""

import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    metrics=json.loads((ROOT/'reports/v5/metrics.json').read_text())
    selection=json.loads((ROOT/'reports/v5/selection.json').read_text())
    verified=json.loads((ROOT/'reports/v5/verification.json').read_text())
    tests=json.loads((ROOT/'reports/v5/tests.json').read_text())
    v4=json.loads((ROOT/'reports/v5/v4_comparison.json').read_text())
    assert verified['offline_full_inference_byte_identical'] and tests['status']=='passed'
    chosen=selection['selected']
    config=chosen['config']
    oof,old,hold=chosen['metrics'],selection['baseline_v3'],metrics['holdout']
    models=[]
    if 'context_config' in chosen:
        models.append('сглаженные обучающие оценки по доменам и заголовкам страниц')
    if config.get('cat_weight',0):
        models.append('второй CatBoost с глубиной 9')
    if config.get('hash_weight',0):
        models.append('хешированные текстовые признаки с IDF по всей обучающей части')
    if config.get('leaf_weight',0):
        models.append('LightGBM')
    errors=lambda m:m['fp']+m['fn']
    selection_flags=' '.join('--include-'+name for name in ('cat','hash')
                             if name in selection.get('included_oof_components',[]))
    hybrid=bool(config.get('hash_weight',0.) and 'context_config' in chosen)
    feedback_path=ROOT/'reports/v5/score_feedback.json'
    public_note=('По сообщению участника public F1 v5 = '+str(json.loads(feedback_path.read_text())['public_f1'])+
                 ', столько же получил v3. Локальное улучшение не дало прироста в таблице турнира.'
                 if feedback_path.exists() else 'Значение public F1 после загрузки v5 пока неизвестно.')
    search_command=('.venv/bin/python experiments/v5/hybrid_search.py' if hybrid
                    else '.venv/bin/python experiments/v5/soft_search.py' if 'context_config' in chosen
                    else '.venv/bin/python experiments/v5/select_ensemble.py '+selection_flags)
    promotion_command=('.venv/bin/python scripts/promote_hybrid_selection.py' if hybrid else
                       '.venv/bin/python scripts/promote_soft_selection.py' if 'context_config' in chosen else '')
    protocol='Параметры моделей выбирались на внутренней проверке development. Каждый OOF прогноз вычислен моделью, которая не обучалась на соответствующем URL. Веса и порог выбраны по этим OOF прогнозам. Дополнительные 178 URL из исправленного CSV добавлены только при финальном обучении.'
    content=f'''# Обученная модель v5

V5 добавляет к v3 {', '.join(models)}. {public_note} Результаты v3 (0.994425) и v4 (0.994096) сообщены участником, доступ к приватной оценке отсутствует. V4 не заменяет v3 как лучший подтвержденный обученный вариант.

## Фактическая проверка

| Проверка | V3 F1 | V4 F1 | V5 F1 | Ошибки v3 / v4 / v5 |
|---|---:|---:|---:|---:|
| Development OOF, 50977 URL | {old['f1']:.9f} | {v4['oof']['f1']:.9f} | {oof['f1']:.9f} | {errors(old)} / {errors(v4['oof'])} / {errors(oof)} |
| Прежний holdout, 12745 URL | 0.989533180 | {v4['holdout']['f1']:.9f} | {hold['f1']:.9f} | 100 / {errors(v4['holdout'])} / {errors(hold)} |

На OOF исправлено {chosen['fixed_baseline_errors']} ошибок v3 и добавлено {chosen['new_errors']} новых. На holdout исправлено {metrics['holdout_fixed_errors']} и добавлено {metrics['holdout_new_errors']} новых. Порог класса 1: {chosen['threshold']:.12f}. F1 на {metrics['unseen_content_rows']} строках holdout без точного совпадения HTML/компактного текста в обучении: {metrics['unseen_content_metrics']['f1']:.9f}.

Прирост небольшой и неравномерный: на одном из трех OOF фолдов ошибок стало меньше, на двух остальных немного больше. На holdout у v4 и v5 одинаковое число ошибок, но разное соотношение FP/FN; F1 v5 немного ниже v4. Новый CSV меняет {metrics['submission']['prediction_changes_vs_v3']} ответов относительно v3.

{protocol}

OOF используется для выбора смеси и не является свежим независимым тестом. Holdout уже применялся в предыдущих экспериментах; это сравнительная диагностика. Новое отдельное обучение с непересекающимися доменами для v5 не проводилось. Локальные метрики не гарантируют роста public или private F1.

## Что это значит

Ансамбль использует URL, текст и статические признаки HTML. Отдельные модели получают разные части этих данных и возвращают оценку вероятности фишинга. Одна модель лучше распознает характерные слова, другая сочетает длину URL, структуру адреса и признаки страницы. Ансамбль усредняет вероятности с заданными весами. Польза новой модели зависит от того, исправляет ли она ошибки прежних моделей.

OOF означает прогнозы для строк, которые конкретная модель не видела при обучении. Обучающая часть делится на 3 группы. Модель учится на двух группах и прогнозирует третью; это повторяется для каждой группы. Полученные прогнозы позволяют выбирать настройки без оценки на тех же примерах, на которых модель обучалась. При этом выбор лучшего варианта среди многих опытов все равно делает OOF оптимистичной оценкой, поэтому нужна отдельная проверка.

Хеширование текста, если его вес ненулевой, переводит слова и последовательности символов в фиксированные числовые столбцы. Это позволяет обработать признаки из всех обучающих строк без огромного словаря. Возможны коллизии, когда разные слова попадают в один столбец. IDF уменьшает вес повсеместно встречающихся признаков и обучается только на обучающей части. Описание алгоритмов: [HashingVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.HashingVectorizer.html), [TfidfTransformer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html).

После смеси используются прежние правила v3: не менее 3 одинаковых страниц с единственной обучающей меткой и не менее 5 URL домена с единственной меткой. Правила строятся заново только по обучающей стороне каждого разбиения. При конфликте признаки не заменяют прогноз. Сам URL, уже встречавшийся в финальном train, получает свою известную обучающую метку. Это дает {verified['known_training_urls']} точных совпадений в test.

Сглаженная контекстная оценка, если она включена в конфигурацию, использует число безопасных и фишинговых примеров с тем же доменом или полным заголовком. К ним добавляются 5 условных примеров с общей обучающей долей классов. Оценка затем сдвигает вероятность базовой модели. Для доменов используется только небольшой чистый обучающий счетчик (1-4 примера), смешанные домены пропускаются. Заголовок должен повториться хотя бы 2 раза. Очень уверенные базовые ответы вне диапазона 0.05-0.95 сохраняются. При применении мягкого контекста решения жестких правил по текущей базовой смеси сохраняются. Параметры контекста: `{json.dumps(chosen.get('context_config',{}),sort_keys=True)}`.

F1 = 2 * TP / (2 * TP + FP + FN). TP - правильно найденный фишинг, FP - безопасный адрес ошибочно назван фишингом, FN - пропущенный фишинг. На высоких значениях F1 несколько дополнительных ошибок уже заметны в таблице турнира.

LightGBM, более глубокий CatBoost и хешированный текст проверены на трех development разбиениях. LightGBM и глубокий CatBoost по отдельности не улучшили полный ансамбль v3. Комбинации новых деревьев, текста и контекста оценены отдельно; в финальную модель входят только компоненты с ненулевым весом. Результаты, включая отрицательные эксперименты, сохранены в `experiments/v5/` и `reports/v5/`. Нормализация и ослабленные правила v4 отклонены после турнирного снижения.

Веса финальной смеси:

До применения обучающих правил и контекста итоговая оценка складывается из шести прогнозов: 35.7% прежний HTML текст, 11.9% новый хешированный текст, 20.4% прежний CatBoost по числовым признакам HTML/URL, 15% новый глубокий CatBoost, 8.5% текстовая модель URL и 8.5% CatBoost по признакам URL. Поля конфигурации ниже задают вложенные смеси; их значения не нужно складывать напрямую.

```json
{json.dumps(config,indent=2)}
```

## Воспроизведение готового CSV

Архив содержит исходники, компактные данные, модель, конфигурацию и результаты проверок. Исходный архив организатора с большим HTML CSV для повторного прогноза не нужен. Команды выполняются из распакованной папки `phishing-hackathon`. Проверенное окружение описано в `reports/v5/environment.json`, зависимости зафиксированы в `requirements.txt`.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/setup_openmp.py
.venv/bin/python -m phishing_url.v5_cli \\
  --model artifacts/v5/ensemble.joblib \\
  --cache data/content_test.joblib \\
  --url-features data/content_url_test.joblib \\
  --sample data/sample.csv \\
  --output submissions/reproduced_content_v5.csv
.venv/bin/python scripts/verify_v5.py
.venv/bin/python -m pytest -q
```

На macOS `setup_openmp.py` проверяет импорт LightGBM. Если библиотека OpenMP не найдена, скрипт связывает wheel LightGBM с уже установленной библиотекой sklearn внутри текущего Python-окружения, сохраняя оригинал для отката. На Linux успешный импорт не требует изменений. Этот шаг проверен на текущем macOS окружении.

Все {tests['tests']} тестов прошли. Сохраненная модель повторно обработала все 16000 URL при запрещенных сетевых соединениях. Полученный CSV побайтово совпал с `submissions/submission_content_v5.csv`, порядок test сохранен. Проверка OOF заново построила обучающие правила на каждом фолде и проверила границы обучения по контрольным суммам. Для деревьев заново проверены все прогнозы выбранных OOF моделей, для хешированного текста - первые 32 записи каждого фолда.

## Повторное обучение и поиск

```bash
.venv/bin/python experiments/v5/screen_catboost.py
.venv/bin/python experiments/v5/screen_leaf.py
.venv/bin/python experiments/v5/screen_hash.py
.venv/bin/python experiments/v5/oof.py --kind leaf
.venv/bin/python experiments/v5/oof.py --kind cat
# Для воспроизведения выбранного ансамбля нужен OOF хешированного текста:
.venv/bin/python experiments/v5/oof.py --kind hash
.venv/bin/python experiments/v5/soft_search.py
.venv/bin/python experiments/v5/select_ensemble.py --include-cat --include-hash
{search_command}
{promotion_command}
.venv/bin/python scripts/train_v5.py --refit-new-models
.venv/bin/python scripts/verify_v5.py
```

`soft_search.py` записывает итог контекста в `reports/v5/soft_selection.json` и `reports/v5/soft_development_oof.npz`. `hybrid_search.py` сравнивает четыре заранее заданные смеси с контекстом и использует зафиксированную четвертую смесь из `experiments/v5/hybrid_base_config.json`. Финальный результат записывается в `hybrid_selection.json`, затем `promote_hybrid_selection.py` переносит его в конфигурацию обучения. `--refit-new-models` обучает добавленные деревья/текст заново; контекстные счетчики всегда строятся заново на development и полном train. Готовые базовые модели v3 также включены; их полное обучение и извлечение HTML из сырого архива описаны в `README_CONTENT.md`. Контрольные суммы не позволяют подменить разбиение или настройки при повторном использовании компонентов. Побайтовая идентичность после нового обучения на другой платформе не гарантируется.

## Граница между моделью и справочным результатом

`submission_reference.csv` получил public F1 1.0 на скриншоте участника, потому что опубликованный исходный набор содержит метки test. Этот способ раскрыт в документации v2/v4. В v5 опубликованные метки test не используются; внешнего справочника в модели и архиве нет. Содержание HTML рассматривается как данные, JavaScript не исполняется. Новых вызовов Astra или субагентов: 0.

Файлы для проверки: `reports/v5/selection.json`, `reports/v5/metrics.json`, `reports/v5/verification.json`, `reports/v5/tests.json` и `reports/v5/tournament_feedback.json`.
'''
    (ROOT/'README_V5.md').write_text(content,encoding='utf-8')
    (ROOT/'reports/v5/RUN_STATUS.md').write_text(
        'Готово и проверено локально. '+public_note+' Приватный F1 неизвестен.\n\n'
        f"Development OOF: {oof['f1']:.9f}; holdout: {hold['f1']:.9f}.\n"
        'CSV: submissions/submission_content_v5.csv. Новых вызовов Astra: 0.\n',encoding='utf-8')
    print('README_V5.md written from verified results.')


if __name__=='__main__':
    main()
