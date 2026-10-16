"""Summarize measured content results and plot actual holdout mistakes."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    out = ROOT / "reports/v3"
    metrics = json.loads((out / "metrics.json").read_text())
    checked = json.loads((out / "verification.json").read_text())
    prior = json.loads((ROOT / "reports/v2/url_model_metrics.json").read_text())
    before, after = prior["previously_used_holdout"], metrics["holdout"]["metrics"]
    oof, domain = metrics["development_oof"], metrics["domain_disjoint"]["metrics"]
    selected = metrics["config"]["selected"]
    known = metrics["submission"]["known_training_urls"]
    rule_matches = metrics["submission"]["training_rule_matches"]
    novelty = metrics["holdout"].get("unseen_content_metrics")
    novel_text = (f"На {novelty['rows']} holdout страницах, чей полный HTML и извлеченное представление не встречались в обучении, F1 = {novelty['f1']:.6f}." if novelty else "В этом разбиении не было пригодного отдельного набора новых страниц для F1.")
    tests_path = out / "tests.json"
    tests = json.loads(tests_path.read_text()) if tests_path.exists() else {"status":"not recorded"}
    text = f"""# Проверенные результаты модели с HTML

HTML улучшил обученный классификатор. Результат турнира пока не известен: файл еще должен загрузить сам участник. Значение 0.995079 из сообщения участника относится к другой, турнирной оценке; его нельзя напрямую сопоставлять с нашим локальным holdout.

| Модель | Набор | Строк | F1 | Ложные тревоги FP | Пропуски FN |
|---|---|---:|---:|---:|---:|
| URL v2 | Прежний holdout | {before['rows']} | {before['f1']:.6f} | {before['fp']} | {before['fn']} |
| URL + HTML | Тот же holdout | {after['rows']} | {after['f1']:.6f} | {after['fp']} | {after['fn']} |
| URL v2 | Development OOF | {prior['development_oof']['rows']} | {prior['development_oof']['f1']:.6f} | {prior['development_oof']['fp']} | {prior['development_oof']['fn']} |
| URL + HTML | Тот же development OOF | {oof['rows']} | {oof['f1']:.6f} | {oof['fp']} | {oof['fn']} |
| URL + HTML | Раздельные домены | {domain['rows']} | {domain['f1']:.6f} | {domain['fp']} | {domain['fn']} |

На одинаковом holdout число ошибок изменилось с {before['fp']+before['fn']} до {after['fp']+after['fn']}. Это повторно используемый сравнительный набор, а не новый независимый тест. Веса и порог выбраны только по development OOF и зафиксированы до расчета holdout F1.

![Ошибки на одинаковом holdout](errors.png)

{novel_text}

Проверка по доменам имеет нулевое пересечение зарегистрированных доменов, включая частные зоны хостинга. Одинаковые страницы на разных доменах могут остаться. Ее показатель служит дополнительной диагностикой, параметры выбраны на том же наборе разработки. Метрика на новых шаблонах в этом разбиении записана отдельно в metrics.json.

## Итоговая модель

Вес текстовой HTML модели внутри HTML ансамбля: {selected['text_weight']}; вклад URL ансамбля: {selected['aux_url_weight']}. Порог класса 1: {selected['threshold']:.12f}. Текстовый классификатор использует TF-IDF четырех представлений и логистическую регрессию C=16. Числовой классификатор использует CatBoost глубины 8, {metrics['config']['numeric_parameters']['iterations']} деревьев, learning_rate 0.045 и l2_leaf_reg 5. URL ансамбль объединяет богатые текстовые признаки и исходный CatBoost.

Точная метка повторяющегося HTML или извлеченного представления используется при минимум {selected['memory_minimum_support']} обучающих подтверждениях. Подсказка по домену требует {selected['domain_minimum_support']} одинаковых меток в обучении и применяется к неуверенному ответу. Противоречивые совпадения оставляют ответ модели. В OOF все справочники построены только по обучающим строкам каждого разбиения.

Финальные модели обучены на 63900 уникальных URL. Дополнительные 178 восстановленных примеров использованы только на этом этапе. В test {known} точных совпадений URL с train и {rule_matches} совпадений с правилами из train. Опубликованные метки test из справочного решения v2 не используются в этой модели или в ее проверках.

## Файлы и проверки

CSV для турнира: ../../submissions/submission_content_v3.csv. Он содержит 16000 строк в исходном порядке test, два столбца Id,Predicted и бинарные значения. Повторный офлайн прогноз совпал побайтово: {checked['byte_identical_replay']}. Правила обычного прогноза дополнительно пересчитаны на всех 50977 OOF строках и совпали с векторизованным поиском.

Автоматические тесты: {tests.get('passed', 'не записано')} пройдено, статус {tests['status']}. Контрольные суммы и подробные численные проверки: verification.json. Источники и контрольные суммы исходного архива: archive_manifest.json. Подробное объяснение и команды: ../../README_CONTENT.md.

Проверялись также текстовые варианты C=4, C=16, NB-преобразование и вариант без URL. Native text CatBoost дал внутренний F1 0.982334 и был исключен до полной проверки. Для него не заявляется holdout или турнирный F1. Все измеренные эксперименты сохранены в experiments/content и experiments/html_astra.
"""
    (out / "RESULTS.md").write_text(text, encoding="utf-8")
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":11})
    fig, ax = plt.subplots(figsize=(7.4,4.4))
    positions = np.arange(2)
    ax.bar(positions-.18, [before['fp'], after['fp']], .36, label="Ложная тревога (FP)", color="#ce7b43")
    ax.bar(positions+.18, [before['fn'], after['fn']], .36, label="Пропуск фишинга (FN)", color="#6272a4")
    ax.set_xticks(positions, ["Только URL, v2", "URL + HTML"])
    ax.set_ylabel("Число ошибочных ответов")
    ax.set_title(f"Один holdout: {after['rows']} URL")
    ax.legend(frameon=False)
    ax.spines[['top','right']].set_visible(False)
    ax.set_ylim(0, max(before['fp'],before['fn'],after['fp'],after['fn'])*1.25)
    for container in ax.containers:
        ax.bar_label(container, padding=3)
    fig.tight_layout()
    fig.savefig(out / "errors.png", dpi=160)
    plt.close(fig)
    print("Saved measured report and error chart")


if __name__ == "__main__":
    main()
