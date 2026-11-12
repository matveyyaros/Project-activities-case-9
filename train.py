import hydra
from omegaconf import DictConfig, OmegaConf
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
from sklearn.compose import ColumnTransformer
import pandas as pd
from project_activities_case_9.data.load_data import load_data

def build_preprocessing(cfg_prep) -> list:
    """Нормализует узел cfg.preprocessing в список шагов (name, estimator).

    Поддерживает:
      * None            -> []
      * DictConfig      -> [(name, instance)]      (один трансформер, в т.ч. ColumnTransformer)
      * ListConfig/list -> [(name, instance), ...] (последовательность шагов)
    """
    if cfg_prep is None:
        return []

    if OmegaConf.is_list(cfg_prep):
        items = list(cfg_prep)
    else:
        items = [cfg_prep]           # одиночный словарь, включая ColumnTransformer

    steps = []
    for i, item in enumerate(items):
        if "_target_" not in item:
            raise ValueError(
                f"preprocessing[{i}] не содержит _target_: {item}"
            )
        cls_name = item._target_.split(".")[-1]
        name = f"prep_{i}_{cls_name}"
        obj = hydra.utils.instantiate(item)

        nodes = [obj]
        for node in nodes:
            if isinstance(node, ColumnTransformer):
                fixed = []
                for t in node.transformers:
                    if isinstance(t, (list, tuple)):
                        if len(t) >= 2:
                            nodes.append(t[1])   # вложенный estimator
                        t = tuple(t)
                    fixed.append(t)
                node.transformers = fixed

            elif isinstance(node, Pipeline):
                for _, step in node.steps:
                    nodes.append(step)
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

    # 6. Сохранение модели
    joblib.dump(pipeline, hydra.core.hydra_config.HydraConfig.get().runtime.output_dir+cfg.output_path)
    print(f"✅ Модель сохранена в {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir+cfg.output_path}")


if __name__ == "__main__":
    main()