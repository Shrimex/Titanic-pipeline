## Описание

Задача классификации — предсказание выживания после столкновения титаника с айсбергом. Проект построен на универсальном ML-пайплайне с акцентом на:
- отсутствие data leakage
- переиспользуемость кода
- сравнение множества моделей через cross-validation
- автоматический поиск гиперпараметров

---

## Структура проекта

```
House_Prices/
├── Data/
│   ├── train.csv
│   └── test.csv
├── src/
│   ├── config.py            # Конфигурация проекта (OmegaConf)
│   ├── data.py              # Загрузка данных
│   ├── EDA.ipynb            # Разведочный анализ данных
│   ├── evaluation.py        # Оценка моделей через CV (F1, accuracy)
│   ├── feature_creation.py  # Feature engineering
│   ├── model_list.py        # Список моделей
│   ├── neural_network.py    # MLP классификатор на PyTorch (sklearn-совместимый)
│   ├── param_grids.py       # Сетки гиперпараметров для GridSearchCV
│   ├── preprocessing.py     # Imputation, encoding, feature engineering pipeline
│   ├── training.py          # Основной пайплайн обучения
│   └── utils.py             # Вспомогательные функции
├── main.py                  # Точка входа
└── requirements.txt
```

---

## Пайплайн

```
Загрузка данных → Feature Engineering → Imputation → Encoding
→ CV оценка всех моделей → GridSearchCV → Финальное обучение → Сабмит
```

### Ключевые решения

## Feature Engineering:

- `Initial` — извлечение титула из имени (Mr / Mrs / Miss / Master / Other) с агрегацией редких значений  
- `Family_Size` — размер семьи (`SibSp + Parch`)  
- `Alone` — бинарный флаг одиночества  
- `Family_Type` — категоризация размера семьи (один / маленькая / средняя / большая)  
- `Has_Cabin` — наличие информации о каюте  
- `Deck` — палуба (первая буква `Cabin`, с обработкой пропусков)  

---

## Imputation (без leakage):

- `Age` — групповая импутация по `Initial` (медиана внутри титула)  
- `Fare` — групповая импутация по `Pclass`  
- `Embarked` — мода (обучается только на train)  
- Все статистики считаются только на train, затем применяются к test  

---

## Encoding:

- `Sex` — бинарное кодирование (`male=0`, `female=1`)  
- `Embarked` — ordinal encoding (`S=0`, `C=1`, `Q=2`)  
- `Initial` — ordinal encoding по социальному статусу  
- `Deck` — ordinal encoding палуб  
- `Age_band` — биннинг возраста  
- `Fare_cat` — биннинг стоимости билета  
- `Is_Child` — бинарный признак (возраст < 16)  

---

## Модели

| Модель | Тип |
|--------|-----|
| DummyClassifier | Baseline |
| Ridge, Lasso, ElasticNet | Линейные |
| DecisionTree | Дерево |
| RandomForest | Ансамбль |
| LightGBM, CatBoost, XGBoost | Градиентный бустинг |
| TitanicNN (MLP) | Нейронная сеть |

Нейронная сеть реализована на PyTorch с sklearn-совместимым интерфейсом, поддерживает настройку оптимизатора (`adam`, `sgd`, `rmsprop`) и scheduler.

---

## Запуск

### Установка зависимостей

```bash
pip install -r requirements.txt
```

### Запуск пайплайна

```bash
python main.py
```

Результаты сохраняются в папку `outputs/`:
- `outputs/metrics/cv_results.csv` — результаты CV по всем моделям
- `outputs/metrics/summary.json` — лучшая модель и параметры
- `outputs/submissions/` — файл для сабмита на Kaggle
- `outputs/models/` — сохранённая модель

---

