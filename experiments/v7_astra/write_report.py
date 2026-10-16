"""Render the measured results; this does not fit or select anything."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(name):
    return json.loads((OUT / name).read_text())


def row(name, metric):
    return (f"| {name} | {metric['f1']:.9f} | {metric['threshold']:.9f} | "
            f"{metric['fp']} | {metric['fn']} | {metric['fp']+metric['fn']} |")


def main():
    inner, oof = read('inner_results.json'), read('oof_results.json')
    verified = read('verification.json')
    execution = [read('inner_execution.json'), read('oof_execution.json')]
    timings = [read('inner_full_vocabulary_fit.json')] + [read(f'fold_{i}_fit.json') for i in range(3)]
    delta = oof['replacement']['f1']-oof['baseline']['f1']
    conclusion = ('Замена повысила OOF F1 при зафиксированных inner-порогах.' if delta > 0 else
                  'Замена не подтвердила прирост OOF F1 при зафиксированных inner-порогах.')
    integration = read('integration_verification.json') if (OUT/'integration_verification.json').exists() else None
    integration_lines = []
    if integration:
        chosen, base = integration['selected'], integration['baseline_v6']
        integration_lines = [
            '## Интеграция в полную смесь v6: результат основной ветки решения', '',
            'Источник: experiments/v7_astra_selection.py и reports/v7/selection.json. '
            'В основной ветке решения выполнена полная замена старого HTML-текста и добавлены '
            'выбранные на inner train-only условные признаки host/path. '
            'Сохранены исходные fold-local правила и контекст. '
            'В этой ветке повторно проверены сохраненные OOF-вероятности, их происхождение, '
            'метрики и разбиения; полный pipeline повторно не запускался.', '',
            '| Модель | F1 | Порог | FP | FN | Ошибок |',
            '|---|---:|---:|---:|---:|---:|',
            row('Baseline v6',base),
            row('Полная замена + hierarchy, выбранный OOF-порог',chosen['metrics']),
            row('Та же замена при прежнем пороге v6',chosen['metrics_at_v6_threshold']), '',
            f"Исправлено {chosen['fixed_errors']} ошибок, добавлено {chosen['new_errors']}. "
            'По fold: ' + ', '.join(f"{r['baseline_errors']} -> {r['candidate_errors']}" for r in chosen['fold_comparison']) + '.',
            'Ограничение: веса смеси и порог выбраны на OOF. Это оптимистичная оценка выбора, '
            'а не независимая оценка обобщения. При старом пороге результат немного хуже baseline; '
            'выигрыш нельзя целиком отнести к новому тексту, поскольку менялись hierarchy и порог. '
            'Holdout/test в этой проверке не читались. Финальные dev/full-компоненты собираются в основной ветке решения.', '',
            'Проверка без обучения и поиска:', '', '```sh', f'cd {ROOT}',
            'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/v7_astra/verify_integration.py',
            'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/v7_astra/write_report.py', '```', '',
        ]
    lines = [
        '# V7: полный словарь TF-IDF', '',
        *integration_lines,
        '## Исходная диагностика: простая смесь текста с CatBoost', '',
        conclusion,
        f"Разница F1 replacement - baseline: {delta:+.9f}. "
        f"Исправлено ошибок: {oof['fixed_errors']}; новых ошибок: {oof['new_errors']}.", '',
        '## Область данных и ограничения', '',
        '- Единственный обучающий источник: data/content_train.joblib, 63900 строк.',
        '- Выбор на inner_train=34663 и inner_validation=6118 через aligned_splits.',
        '- OOF: original development=50977, StratifiedKFold(3, shuffle=True, random_state=2026).',
        '- Holdout и 178 дополнительных строк не использовались для fit/predict/выбора. '
        'Монолитный разрешенный train-cache загружается целиком; aligned_splits проверяет соответствие старому train.',
        '- external/, reports/v2/, reference submission, test-метки и опубликованный датасет не читались; '
        'сетевых запросов нет. Audit hook запрещает эти входы и запись вне experiments/v7_astra/.',
        '- OOF не является вложенной независимой оценкой выбора модели: inner находится внутри development. '
        'Holdout не оценивался. Baseline v6 ранее выбиралась на development OOF и дана только для контекста.', '',
        '## Кандидат 1: полный fit при исходных лимитах', '',
        'FullVocabularyTextModel наследует представления и logistic regression из ContentTextModel; '
        'изменено только обучение словаря/IDF на ВСЕХ fit-строках. На каждом fold создаются новые vectorizers.', '',
        '- C=16, liblinear, nb=False, min_df=2, sublinear_tf=True, float32, max_iter=500, seed=20261006.',
        '- word: title дважды + text[:12000], word (1,2), максимум 140000 признаков.',
        '- char: title + text[:2000], char_wb (3,5), максимум 100000.',
        '- markup[:16000]: word (1,2), максимум 70000; URL[:4096]: char (3,5), максимум 90000.',
        '- Итого 400000 признаков. Сохранены исходные ограничения текста и признаков.',
        '- Fits последовательные; ограничения BLAS/OpenMP и threadpool_limits=2. '
        'Обнаруженный OpenMP pool проверен: 2 потока.', '',
        '## Протокол выбора', '',
        'До первого fit сохранен preregistered.json: один estimator; '
        'baseline = 0.5*старый текст + 0.5*cat_inner_1; '
        'добавление нового текста к baseline с весами 0.10/0.25/0.50; '
        'отдельно заранее предусмотрена замена 0.5*новый текст + 0.5*CatBoost.',
        'Первичный критерий добавления не прошел: все три смеси хуже baseline. '
        'Заранее предусмотренная вторичная замена улучшила inner на одну суммарную ошибку. '
        'После просмотра inner принято явно отмеченное исследовательское решение проверить эту замену на OOF. '
        'Это изменение решения о запуске, а не успешное прохождение первоначального критерия.',
        'Конфиг, веса 0.5/0.5 и порог замены зафиксированы в oof_selection.json до обучения OOF. '
        'В этой отдельной диагностике новых параметров, перебора весов или подбора порога на OOF не было. '
        'Последующая интеграция в полную v6 описана отдельно выше.', '',
        '## Inner-validation', '',
        'Пороги в этой таблице подобраны на inner; оценки используются для выбора, не как независимый результат.', '',
        '| Модель | F1 | Порог | FP | FN | Ошибок |',
        '|---|---:|---:|---:|---:|---:|',
        row('Старый текст',inner['old_text']['tuned']),
        row('Полный словарь',inner['new_text']['tuned']),
        row('Baseline: старый текст + CatBoost',inner['baseline']['tuned']),
        *[row(f"Добавление, вес {r['weight']}",r['tuned']) for r in inner['blends']],
        row('Замена: новый текст + CatBoost',inner['replace_text_half_cat']['tuned']), '',
        '## Original development OOF', '',
        'Пороги только из inner. Полное покрытие 50977 строк ровно по одному разу; '
        'fold assignments обеих сохраненных baseline-компонент совпали.', '',
        '| Модель | F1 | Порог | FP | FN | Ошибок |',
        '|---|---:|---:|---:|---:|---:|',
        row('Старый текст',oof['old_text']),row('Полный словарь',oof['standalone']),
        row('Baseline: старый текст + CatBoost',oof['baseline']),row('Замена: новый текст + CatBoost',oof['replacement']), '',
        'При едином пороге 0.5:', '',
        '| Модель | F1 | Порог | FP | FN | Ошибок |',
        '|---|---:|---:|---:|---:|---:|',
        row('Старый текст',oof['old_text_at_05']),row('Полный словарь',oof['standalone_at_05']),
        row('Baseline',oof['baseline_at_05']),row('Замена',oof['replacement_at_05']),
        row('Существующая v6 (только контекст)',oof['v6_at_05']), '',
        '| Fold | Baseline F1 | Замена F1 | Разница |', '|---|---:|---:|---:|',
        *[f"| {r['fold']} | {r['baseline']['f1']:.9f} | {r['replacement']['f1']:.9f} | "
          f"{r['replacement']['f1']-r['baseline']['f1']:+.9f} |" for r in oof['folds']], '',
        '## Измеренное время и проверки', '',
        '| Fit | Обучение, с | Прогноз, с | Обучение + прогноз + сохранение/проверки, с |',
        '|---|---:|---:|---:|',
        *[f"| {r['name']} | {r['fit_seconds']:.2f} | {r['predict_seconds']:.2f} | {r['total_seconds']:.2f} |" for r in timings], '',
        f"Суммарное время процессов inner + OOF: {sum(r['seconds'] for r in execution):.2f} с; "
        f"отдельная verification: {verified['seconds']:.2f} с. "
        f"Максимальный ru_maxrss (macOS): {max(r['peak_process_rss_bytes'] for r in timings)/1024**3:.2f} GiB.",
        '- Все 4 модели сохранены и загружены с guard; для каждой прогнозы совпали побитно на 32 реальных validation-строках. '
        'Неверный training guard отвергнут.',
        '- Для 32 word-признаков независимо пересчитаны document frequencies на всех 34663 inner_train строках; '
        f"IDF совпал с максимальной абсолютной погрешностью {verified['idf_max_abs_error']:.3g}.",
        'При прежних лимитах изменился состав словарей: новых word-признаков '
        f"{verified['vocabulary_comparison']['word']['new_only']}, char "
        f"{verified['vocabulary_comparison']['char']['new_only']}, markup "
        f"{verified['vocabulary_comparison']['markup']['new_only']}, URL "
        f"{verified['vocabulary_comparison']['url']['new_only']}. "
        'Число признаков каждого представления осталось прежним.',
        '- Проверены параметры sklearn clone, SHA256 исходников и данных, guards, split indices, '
        'fold coverage, baseline alignment и повторный расчет сохраненных метрик.',
        '- В models/ сохранены четыре модели; guards/ содержит точные training guards; '
        'inner_full_vocabulary.npz и fold_0/1/2.npz содержат train/validation indices, labels и probabilities. '
        'development_oof.npz содержит текстовый OOF, replacement, baseline и folds.', '',
        '## Воспроизведение', '',
        'Запуск из корня проекта. Команды записывают только experiments/v7_astra/. '
        'run_oof.py проверяет неизменность источников и зафиксированного решения. '
        'run.py oof относится к первоначальному не прошедшему критерию добавления; '
        'для фактически проведенной проверки замены нужен run_oof.py.', '',
        '```sh', f'cd {ROOT}',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u experiments/v7_astra/run.py inner > experiments/v7_astra/inner.log 2>&1',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u experiments/v7_astra/run_oof.py > experiments/v7_astra/oof.log 2>&1',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/v7_astra/verify.py',
        'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/v7_astra/write_report.py', '```', '',
        'Точные конфигурации, пороги, полные метрики, времена и версии библиотек лежат в JSON рядом с этим отчетом. '
        'Никакой production/submission-модели не изменено.', ''
    ]
    lines.extend(['## Передача текстового OOF в v6', '',
        'development_oof.npz содержит ЧИСТЫЕ вероятности нового текста в probabilities, '
        'cache indices в indices, исходные train-метки в labels и номера folds. '
        'Колонка replacement относится только к диагностической смеси с CatBoost. '
        'Точное замещение HTML-текста внутри финальной v6 с сохранением ее правил/контекста '
        'выполнено в основной ветке решения; проверенный результат отражен в разделе интеграции. '
        'Сравнение с v6_at_05 не заменяет такую проверку.', ''])
    expanded_path=OUT/'expanded/inner_results.json'
    if expanded_path.exists():
        expanded=json.loads(expanded_path.read_text())
        selected=expanded['selection']
        elapsed=json.loads((OUT/'expanded/inner_execution.json').read_text())['seconds']
        lines.extend(['## Кандидат 2: расширенные лимиты по указанию пользователя', '',
            'Единственный дополнительный конфиг: word=280000, char=200000, markup=140000, '
            'url=180000, C=16; остальные настройки первого кандидата. '
            'Зафиксирован до fit в expanded/preregistered.json. '
            'Нового поиска весов нет: сравнивается замена 0.5*текст + 0.5*CatBoost.',
            'Критерий запуска OOF: минимум 2 neterrors лучше baseline с 52 ошибками '
            'и положительный прирост F1 на inner.', '',
            '| Модель | F1 | Порог | FP | FN | Ошибок |',
            '|---|---:|---:|---:|---:|---:|',
            row('Baseline inner',expanded['baseline']['tuned']),
            row('Расширенный текст',expanded['new_text']['tuned']),
            row('Замена расширенным текстом',expanded['replacement']['tuned']), '',
            f"Inner neterrors gain: {selected['inner_net_errors_gain']:+d}. "
            f"Время inner-процесса: {elapsed:.2f} с. "
            f"Критерий OOF: {'пройден' if selected['run_oof'] else 'не пройден'}.", ''])
        extra_oof=OUT/'expanded/oof_results.json'
        if extra_oof.exists():
            eo=json.loads(extra_oof.read_text())
            eo_seconds=json.loads((OUT/'expanded/oof_execution.json').read_text())['seconds']
            lines.extend(['OOF второго кандидата при замороженном inner-пороге:', '',
                '| Модель | F1 | Порог | FP | FN | Ошибок |',
                '|---|---:|---:|---:|---:|',
                row('Baseline',eo['baseline']),row('Расширенный текст',eo['standalone']),
                row('Замена',eo['replacement']), '',f'Время OOF-процесса: {eo_seconds:.2f} с.', ''])
        else:
            lines.extend(['OOF второго кандидата не запускался: порог inner-пользы не достигнут.', ''])
        lines.extend(['Воспроизведение дополнительной проверки:', '', '```sh',f'cd {ROOT}',
            'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u experiments/v7_astra/expanded.py inner > experiments/v7_astra/expanded_inner.log 2>&1',
            'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -u experiments/v7_astra/expanded.py oof > experiments/v7_astra/expanded_oof.log 2>&1',
            'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/v7_astra/verify_expanded.py',
            'PYTHONDONTWRITEBYTECODE=1 .venv/bin/python experiments/v7_astra/write_report.py',
            '```', ''])
    (OUT/'README.md').write_text('\n'.join(lines))
    print(OUT/'README.md')


if __name__ == '__main__':
    main()
