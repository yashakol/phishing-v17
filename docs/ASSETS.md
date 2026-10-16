# Состав данных и весов

| Файл после восстановления | Размер, байт | SHA256 |
| --- | ---: | --- |
| `data/test.csv` | 1036124 | `e496f56c793d09aab88814891ab7524d9785dae8e90025e47803b2054b7448f5` |
| `data/sample.csv` | 116903 | `df29e85ebe7971ebbc00e1f653b85c77b0ef1169bf4cb03950e287e891779f48` |
| `data/train.csv` | 4282337 | `c1a9ace6c32cf5afa65cc0c88b73c2865009bd8c7cf48bdefc78a9016ea2f70c` |
| `artifacts/v17/ensemble.joblib` | 107039345 | `61691818414b6a28edf38f15f4bc2077b2e62803efdb6a08b19aa58bf3ae0c48` |
| `reports/v17/full_training_frame.npz` | 3060870 | `c3fec127c655ae9959606a31ddc6992cd46501b86be46bb89911c23154cbaa64` |
| `reports/v17/expert_fit.json` | 24802 | `665e2b060e3d3a62bfbbd3f03d274e6ca540624dfa308795967e90df753548c6` |
| `reports/v17/preregistered.json` | 2670 | `38e7e4aee9619f436a2365500b8a2629bbbf17aca0290e0cd2ada7d7f15eb167` |
| `data/content_test.joblib` | 58290133 | `30a4c3bfa4bdb3a9f2321d73b92fadbf44a28f3a13837007caaa98bb49ec2231` |
| `data/extra_html_test.joblib` | 60021554 | `22d3ca573f468195f2659ba06fdbab02dc8cd5f8d330fc17f872de847cc4fc05` |
| `data/content_url_test.joblib` | 1945232 | `75c49c633bff5db1cae0de781511c50245034c9488d53f0e5f10f561e54095cc` |
| `data/v17_training_rows.csv` | 424836 | `2a075f82036e3d12362a4e27b224c426b21cfd77b919e00edf5442e3f4c32542` |

Файлы входят в Git через `payload/parts/*.bin`. Каждая часть не превышает 16 MiB. Восстановленные копии находятся в `.runtime/`, который исключен из Git. Фильтры Git LFS и отдельное скачивание моделей не требуются.

Ожидаемый файл ответа находится в `expected/`. Это результат конкретной отправки, а не набор истинных тестовых меток для обучения V17. Для reference его происхождение из опубликованного справочника раскрыто отдельно.
