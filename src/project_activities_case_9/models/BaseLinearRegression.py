from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def build_model(**params):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearRegression())
    ])
    if params:
        pipe.set_params(**params)
    return pipe