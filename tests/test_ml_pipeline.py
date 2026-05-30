import tempfile
import unittest
import zipfile
from pathlib import Path

from tennis_ml.preprocessing import load_and_preprocess_tennis_data, preprocess_rows
from tennis_ml.training import save_model, train_mean_regressor
from tennis_ml.usage import load_model, predict


class TestMlPipeline(unittest.TestCase):
    @staticmethod
    def _sample_csv_data() -> str:
        return """Tournament,Series,Court,Surface,Round,Winner,Player_1,Player_2,Score,Date,Rank_1,Rank_2,Pts_1,Pts_2,Odd_1,Odd_2,Best of
Open,A,Outdoor,Hard,R32,Rafael Nadal,Rafael Nadal,Roger Federer,6-4 6-4,2024-01-10,1,2,10000,9000,1.5,2.5,3
Open,A,Outdoor,Hard,R16,Roger Federer,Rafael Nadal,Roger Federer,4-6 6-2 6-1,2024-01-12,-1,2,-1,9000,-1.0,2.0,3
Masters,B,Indoor,Clay,QF,Rafael Nadal,Rafael Nadal,Novak Djokovic,7-6 6-7 7-5,2024-02-02,1,3,10000,8500,1.8,2.1,3
Masters,B,Indoor,Clay,SF,Novak Djokovic,Rafael Nadal,Novak Djokovic,3-6 6-4 6-4,2024-02-05,1,3,10000,8500,1.9,2.2,3
Open,A,Outdoor,Hard,F,Rafael Nadal,Rafael Nadal,Carlos Alcaraz,6-3 6-3,2024-03-01,1,4,10000,8000,1.6,2.4,3
"""

    def test_preprocess_skips_incomplete_rows(self) -> None:
        rows = [
            {"speed": "100", "spin": "2000", "result": "1"},
            {"speed": "", "spin": "1800", "result": "0"},
        ]

        features, targets = preprocess_rows(rows, ["speed", "spin"], "result")

        self.assertEqual(features, [[100.0, 2000.0]])
        self.assertEqual(targets, [1.0])

    def test_load_and_preprocess_tennis_data(self) -> None:
        csv_data = self._sample_csv_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "atp_tennis.csv"
            file_path.write_text(csv_data, encoding="utf-8")

            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(
                file_path
            )

        self.assertEqual(X_train.shape[0], 4)
        self.assertEqual(X_test.shape[0], 1)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 1)
        self.assertEqual(X_train.shape[1], len(feature_names))
        self.assertIn("Player_1", feature_names)
        self.assertIn("Player_2", feature_names)

    def test_load_and_preprocess_tennis_data_from_zip_member_path(self) -> None:
        csv_data = self._sample_csv_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            archive_path = Path(tmp_dir) / "archive.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr("atp_tennis.csv", csv_data)

            dataset_path = f"{archive_path}/atp_tennis.csv"
            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(
                dataset_path
            )

        self.assertEqual(X_train.shape[0], 4)
        self.assertEqual(X_test.shape[0], 1)
        self.assertEqual(len(y_train), 4)
        self.assertEqual(len(y_test), 1)
        self.assertEqual(X_train.shape[1], len(feature_names))

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
