# Работа с Big Data

Веб-приложение для загрузки больших данных, визуального анализа и обучения ML-моделей.  
Выполнили: Грищук Елизавета, Красноперова Елизавета, Лаузер Янина

---

## Функциональность

- **Загрузка данных** — CSV/TSV файлы до 500 МБ с прогресс-баром
- **Обзор датасета** — превью таблицы, типы колонок, статистики, пропуски
- **Визуализация** — гистограмма, scatter, корреляционная матрица, boxplot, missing values
- **Обучение моделей** — Logistic/Linear Regression и Random Forest (авто-определение задачи: классификация или регрессия)
- **Результаты** — метрики (Accuracy, F1, R², RMSE), confusion matrix / actual vs predicted, feature importances

---

## Стек

| Компонент | Технология |
|-----------|-----------|
| Backend   | Python 3.10+, Flask |
| ML        | scikit-learn |
| Визуализация | matplotlib, seaborn |
| Frontend  | Vanilla JS + CSS (тёмная тема) |

---

## Запуск

### 1. Клонировать репозиторий
```bash
git clone <url>
cd bigdata-lab
```

### 2. Установить зависимости
```bash
pip install -r requirements.txt
```

### 3. Запустить приложение
```bash
python app.py
```

Открыть в браузере: **http://localhost:5000**

---

## Структура проекта

```
bigdata-lab/
├── app.py              # Flask-приложение (API + логика)
├── requirements.txt    # Зависимости
├── README.md
├── templates/
│   └── index.html      # Веб-интерфейс (SPA)
└── uploads/            # Загруженные файлы (создаётся автоматически)
```

---

## API эндпоинты

| Метод | Путь | Описание |
|-------|------|----------|
| `POST` | `/api/upload` | Загрузка CSV/TSV файла |
| `POST` | `/api/visualize` | Построение графика |
| `POST` | `/api/train` | Обучение модели |
| `GET`  | `/api/status` | Статус: загружены ли данные, обучена ли модель |

### Пример: загрузка файла
```bash
curl -X POST http://localhost:5000/api/upload -F "file=@data.csv"
```

### Пример: визуализация
```bash
curl -X POST http://localhost:5000/api/visualize \
  -H "Content-Type: application/json" \
  -d '{"chart_type": "histogram", "col_x": "age"}'
```

Типы графиков: `histogram`, `scatter`, `correlation`, `boxplot`, `missing`

### Пример: обучение модели
```bash
curl -X POST http://localhost:5000/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "target": "label",
    "features": ["age", "salary", "score"],
    "model": "rf",
    "test_size": 0.2
  }'
```

Модели: `auto`, `logistic`, `linear`, `rf`

---

## Демо

Тестовый датасет можно сгенерировать:
```python
import pandas as pd, numpy as np
np.random.seed(42)
n = 10000
df = pd.DataFrame({
    'age': np.random.randint(18, 70, n),
    'salary': np.random.randint(30000, 200000, n),
    'experience': np.random.randint(0, 40, n),
    'department': np.random.choice(['IT','HR','Sales','Finance'], n),
    'score': np.random.uniform(0, 1, n).round(3),
    'target': np.random.choice([0, 1], n),
})
df.to_csv('test_data.csv', index=False)
```

---

