"""
Runs the model on new input data.
"""

import parse
import model
import pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def main():
    data = pickle.load(open("audio.pkl", "rb"))

    x = data.loc[:2000, 'audio']
    y = data.loc[:2000, 'id']
    
    X_train, X_test, y_train, y_test = parse.window_data(x, y, 0.5, 0.5, 16000, 0.25)
    print(X_train.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    print(np.unique(y_test))

    rfc = RandomForestClassifier()
    pipeline = model.generate_pipeline(rfc)

    pipeline.fit(X_train, y_train)
    print(pipeline.score(X_test, y_test))




if __name__ == '__main__':
    main()
