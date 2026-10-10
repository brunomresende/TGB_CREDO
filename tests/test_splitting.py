import numpy as np

from credal_cp_tgb.data.splitting import split_train_calibration_test


def test_paper_split_sizes():
    X = np.arange(1000)[:, None]
    y = np.arange(1000)
    split = split_train_calibration_test(X, y, random_state=1)
    assert len(split.X_train) == 560
    assert len(split.X_calibration) == 240
    assert len(split.X_test) == 200

