Для корректной работы ноутбуков нужно добавить в начало первой ячейки 
```
import os
os.sys.path.append('../scripts/')
```
А также скачать следующие данные https://disk.360.yandex.ru/d/wZwTmWQpJklTyg, https://disk.360.yandex.ru/d/fhh4bGnE7DPwDg и положить их в data/ 

## Краткое описание ноутбуков

- `experiments/BigLeafes.ipynb` — анализ «тяжёлых» листьев MST и влияния отдельных токенов (в т.ч. `not`).
- `experiments/BigLeafesAndDiff.ipynb` — ACF/PACF-анализ зависимости между позицией токена и его весом.
- `experiments/CalculationPHDimScale.ipynb` — базовый расчёт квантильных оценок PH-dimension по корпусу.
- `experiments/CalculationPHDimScaleContinue.ipynb` — продолжение анализа результатов базового расчёта PH-dimension.
- `experiments/CalculationPHDimScaleContinue-Copy1.ipynb` — сравнение квантильного PH-dimension между разными моделями.
- `experiments/CalculationPHDimScaleContinue-Copy2.ipynb` — альтернативный прогон продолжения с изменёнными настройками.
- `experiments/ClassificationTokens.ipynb` — связь нерегулярных токенов с графовыми метриками и классификацией токенов.
- `experiments/EvolutionMSTbyDifferentLayers.ipynb` — эволюция структуры MST по слоям модели.
- `experiments/EvolutionPOSTagging.ipynb` — POS-анализ в контексте графовых характеристик токенов.
- `experiments/EvolutionPOSTaggingUsingDifferentLayers.ipynb` — послойное сравнение POS-паттернов.
- `experiments/HallucinationDetection.ipynb` — детекция галлюцинаций для коротких ответов.
- `experiments/HalluNew.ipynb` — анализ галлюцинаций в длинных ответах.
- `experiments/MSTConsistency.ipynb` — проверка устойчивости MST при сокращении/изменении текста.
- `experiments/PHDEvolution.ipynb` — связь распределений POS-тегов с квантилями PH-dimension.
- `experiments/PracticeExperimentsWithScaledPHDim.ipynb` — ранние exploratory-эксперименты со scaled PH-dimension.
- `experiments/ProbabiltyExplainer.ipynb` — интерпретация факторов, влияющих на итоговые весовые метрики текста.
- `experiments/TFIDF.ipynb` — вспомогательные TF-IDF проверки и baseline-сравнения.
- `experiments/YanaSaveScript.ipynb` — сервисный ноутбук для сохранения рёбер графа в артефакты.
