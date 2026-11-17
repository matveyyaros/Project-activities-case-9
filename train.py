import hydra
from omegaconf import DictConfig, OmegaConf
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, cross_validate,KFold,cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

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
    df = df[df["Row#"] < 759]
    X= df[cfg.data.numerical_features]
    y=df[cfg.data.target_feature]
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
    
    # 3. Обучение
    cv = KFold(n_splits=cfg.cv.folds, shuffle=True, random_state=cfg.seed)
    scoring = cfg.cv.scoring
    model=pipeline
    if "grid_param" in cfg:
        param_grid = OmegaConf.to_container(cfg.grid_param, resolve=True)
        grid = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid["hyperparams"],
            cv=cv,
            scoring=scoring,
            n_jobs=param_grid["n_jobs"],
            refit=True,
            return_train_score=True,
            error_score="raise",
            verbose=1,
        )
        grid.fit(X_train, y_train)
        model=grid.best_estimator_
    else:
        model.fit(X_train, y_train)
    # TODO
    # 5. Оценка на тесте
    y_pred = model.predict(X_test)
    report_output=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir+cfg.output_path
    generate_report(y_true=y_test,y_pred=y_pred,cfg=cfg,model=model,output_dir=report_output,grid=grid)

    # 6. Сохранение модели
    model_output=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir+cfg.output_path
    joblib.dump(model, model_output)
    print(f"✅ Модель сохранена в {model_output}")


if __name__ == "__main__":
    main()