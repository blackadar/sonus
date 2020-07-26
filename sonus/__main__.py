"""
Runs the model on new input data.
"""

import sonus.parse as parse
import sonus.pipeline as pipeline
import pickle
from sklearn.ensemble import RandomForestClassifier


def train():
    # Load the pickle containing wav data, previously processed.
    data = pickle.load(open("audio.pkl", "rb"))
    # Take the first N samples.
    N = 2000
    x = data.loc[:N, 'audio']
    y = data.loc[:N, 'id']

    # Window the audio samples
    X_train, X_test, y_train, y_test = parse.window_data(x, y, 0.5, 0.5, 16000, 0.25)

    # Compute the FFTs
    X_train_fft = pipeline.fft(X_train)
    X_test_fft = pipeline.fft(X_test)

    # Compute the statistics on the audio data
    X_train_stat = pipeline.statistics(X_train)
    X_test_stat = pipeline.statistics(X_test)

    # Compute the statistics on the FFT
    X_train_fft_stat = pipeline.statistics(X_train_fft)
    X_test_fft_stat = pipeline.statistics(X_test_fft)
    X_train_fft_arg_stat = pipeline.arg_statistics(X_train_fft)
    X_test_fft_arg_stat = pipeline.arg_statistics(X_test_fft)

    # Assemble the features
    X_train_total = pipeline.column_join(X_train_fft, X_train_stat, X_train_fft_stat, X_train_fft_arg_stat)
    X_test_total = pipeline.column_join(X_test_fft, X_test_stat, X_test_fft_stat, X_test_fft_arg_stat)

    # Create the model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=1000, n_jobs=4)  # WARNING: n_jobs sets parallelized processes.

    # Create the pipeline, with the model
    pl = pipeline.generate_pipeline(model)
    pl.fit(X_train_total, y_train)

    # Print the model performance on test and training data.
    print()
    print(f"Test Accuracy: {pl.score(X_test_total, y_test)}")
    print(f"Train Accuracy: {pl.score(X_train_total, y_train)}")


def predict():
    # TODO: Read Model from Save
    pass


if __name__ == '__main__':
    train()
