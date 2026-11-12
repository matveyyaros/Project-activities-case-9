import hydra
from omegaconf import DictConfig, OmegaConf
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
import pandas as pd

from project_activities_case_9.data.load_data import load_data

from project_activities_case_9.evaluation.GenerateReport import generate_report
def build_preprocessing(cfg_prep) -> list:
    """Собирает шаги препроцессинга из cfg.preprocessing.

    Требует, чтобы в YAML у каждого _target_-узла стоял `_convert_: all` —
    тогда Hydra рекурсивно приводит вложенные ListConfig/DictConfig
    к нативным list/dict, и sklearn получает корректные списки колонок.
    """
    if cfg_prep is None:
        return []

    items = list(cfg_prep) if OmegaConf.is_list(cfg_prep) else [cfg_prep]

    steps = []
    for i, item in enumerate(items):
        if "_target_" not in item:
            raise ValueError(f"preprocessing[{i}] не содержит _target_: {item}")
        cls_name = item._target_.split(".")[-1]
        name = f"prep_{i}_{cls_name}"
        steps.append((name, hydra.utils.instantiate(item)))

    return steps

@hydra.main(config_path="./conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    # 1. Данные
    load_data()
    df= pd.read_csv("data/raw/saurabhshahane/wild-blueberry-yield-prediction/Data in Brief/Data in Brief/WildBlueberryPollinationSimulationData.csv")
    X= df.drop("yield",axis=1)
    y=df["yield"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        random_state=cfg.seed,
    )

    # 2. Пайплайн: препроцессинг (0..N шагов) + модель
    steps = build_preprocessing(cfg.preprocessing)
    steps.append(("model", hydra.utils.instantiate(cfg.model)))
    pipeline = Pipeline(steps)
    print("Pipeline steps:", [name for name, _ in pipeline.steps])
 
    # 3. Опциональная кросс-валидация на train
    if cfg.cv.enabled:
        scores = cross_val_score(pipeline, X_train, y_train, cv=cfg.cv.folds)
        cv_scores=scores
        print(f"CV accuracy: {scores.mean():.4f} ± {scores.std():.4f}")

    # 4. Обучение
    pipeline.fit(X_train, y_train)
    # TODO
    # 5. Оценка на тесте
    y_pred = pipeline.predict(X_test)
    report_output=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir+cfg.output_path
    generate_report(y_true=y_test,y_pred=y_pred,cfg=cfg,model=pipeline,output_dir=report_output)

    # 6. Сохранение модели
    model_output=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir+cfg.output_path
    joblib.dump(pipeline, model_output)
    print(f"✅ Модель сохранена в {model_output}")


if __name__ == "__main__":
    main()