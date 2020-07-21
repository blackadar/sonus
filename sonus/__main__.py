"""
Runs the model on new input data.
"""

import sonus.parse as parse
import sonus.pipeline as pipeline
import pickle
from sklearn.ensemble import RandomForestClassifier


def train():
    data = pickle.load(open("audio.pkl", "rb"))
    x = data.loc[:2000, 'audio']
    y = data.loc[:2000, 'id']

    X_train, X_test, y_train, y_test = parse.window_data(x, y, 0.5, 0.5, 16000, 0.25)

    X_train_fft = pipeline.fft(X_train)
    X_test_fft = pipeline.fft(X_test)

    X_train = pipeline.column_join(X_train, X_train_fft)
    X_test = pipeline.column_join(X_test, X_test_fft)

    model = RandomForestClassifier()
    pl = pipeline.generate_pipeline(model)

    pl.fit(X_train, y_train)
    print(pl.score(X_test, y_test))


def predict():
    # TODO: Read Model from Save
    pass


if __name__ == '__main__':
    train()
