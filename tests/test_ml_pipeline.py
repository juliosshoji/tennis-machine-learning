import tempfile
import unittest
from pathlib import Path

from tennis_ml.preprocessing import preprocess_rows
from tennis_ml.training import save_model, train_mean_regressor
from tennis_ml.usage import load_model, predict


class TestMlPipeline(unittest.TestCase):
    def test_preprocess_skips_incomplete_rows(self) -> None:
        rows = [
            {"speed": "100", "spin": "2000", "result": "1"},
            {"speed": "", "spin": "1800", "result": "0"},
        ]

        features, targets = preprocess_rows(rows, ["speed", "spin"], "result")

        self.assertEqual(features, [[100.0, 2000.0]])
        self.assertEqual(targets, [1.0])

    def test_train_save_load_and_predict(self) -> None:
        features = [[100.0, 2000.0], [110.0, 2100.0], [105.0, 1900.0]]
        targets = [1.0, 0.0, 1.0]

        model = train_mean_regressor(features, targets)

        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "model.json"
            save_model(model, model_path)
            loaded_model = load_model(model_path)

        predictions = predict(loaded_model, [[98.0, 1800.0], [108.0, 2050.0]])

        self.assertEqual(predictions, [2.0 / 3.0, 2.0 / 3.0])


if __name__ == "__main__":
    unittest.main()
