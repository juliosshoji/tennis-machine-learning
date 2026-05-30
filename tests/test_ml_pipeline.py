import tempfile
import unittest
import zipfile
import json
from pathlib import Path

from tennis_ml.__main__ import main as entrypoint_main
from tennis_ml.preprocessing import load_and_preprocess_tennis_data, preprocess_rows
from tennis_ml.training import (
    save_model,
    save_model_json,
    train_mean_regressor,
    train_decision_tree,
    train_neural_network,
    train_random_forest,
    train_kmeans_clustering,
    evaluate_model,
    get_feature_importance,
)
from tennis_ml.usage import load_model, predict, predict_proba, predict_mean_regressor


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

    def test_train_decision_tree(self) -> None:
        """Test decision tree training and evaluation."""
        csv_data = self._sample_csv_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "atp_tennis.csv"
            file_path.write_text(csv_data, encoding="utf-8")

            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(
                file_path
            )

            dt_model = train_decision_tree(X_train, y_train)
            metrics = evaluate_model(dt_model, X_test, y_test)

        self.assertIn("accuracy", metrics)
        self.assertIn("precision", metrics)
        self.assertIn("recall", metrics)
        self.assertIn("f1", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_train_neural_network(self) -> None:
        """Test neural network training and evaluation."""
        csv_data = self._sample_csv_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "atp_tennis.csv"
            file_path.write_text(csv_data, encoding="utf-8")

            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(
                file_path
            )

            nn_model = train_neural_network(X_train, y_train)
            metrics = evaluate_model(nn_model, X_test, y_test)

        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)
        self.assertLessEqual(metrics["accuracy"], 1)

    def test_train_random_forest(self) -> None:
        """Test random forest training and evaluation."""
        csv_data = self._sample_csv_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "atp_tennis.csv"
            file_path.write_text(csv_data, encoding="utf-8")

            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(
                file_path
            )

            rf_model = train_random_forest(X_train, y_train)
            metrics = evaluate_model(rf_model, X_test, y_test)

        self.assertIn("accuracy", metrics)
        self.assertGreaterEqual(metrics["accuracy"], 0)

    def test_feature_importance_extraction(self) -> None:
        """Test feature importance extraction from models."""
        csv_data = self._sample_csv_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "atp_tennis.csv"
            file_path.write_text(csv_data, encoding="utf-8")

            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(
                file_path
            )

            # Test with Decision Tree
            dt_model = train_decision_tree(X_train, y_train)
            importance = get_feature_importance(dt_model, feature_names, top_n=5)

        self.assertGreater(len(importance), 0)
        # Check that it returns tuples of (feature_name, importance_score)
        for feature_name, score in importance:
            self.assertIsInstance(feature_name, str)
            self.assertIsInstance(score, float)

    def test_kmeans_clustering(self) -> None:
        """Test K-means clustering training."""
        csv_data = self._sample_csv_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "atp_tennis.csv"
            file_path.write_text(csv_data, encoding="utf-8")

            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(
                file_path
            )

            kmeans_model = train_kmeans_clustering(X_train, n_clusters=2)

        self.assertEqual(kmeans_model.n_clusters, 2)
        self.assertGreater(len(kmeans_model.cluster_centers_), 0)

    def test_save_load_pickle_model(self) -> None:
        """Test saving and loading models with pickle."""
        csv_data = self._sample_csv_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "atp_tennis.csv"
            file_path.write_text(csv_data, encoding="utf-8")

            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(
                file_path
            )

            dt_model = train_decision_tree(X_train, y_train)
            model_path = Path(tmp_dir) / "model.pkl"
            save_model(dt_model, model_path)

            loaded_model = load_model(model_path)
            predictions = predict(loaded_model, X_test)

        self.assertEqual(len(predictions), X_test.shape[0])
        for pred in predictions:
            self.assertIn(pred, [0.0, 1.0])

    def test_predict_proba(self) -> None:
        """Test probability predictions."""
        csv_data = self._sample_csv_data()

        with tempfile.TemporaryDirectory() as tmp_dir:
            file_path = Path(tmp_dir) / "atp_tennis.csv"
            file_path.write_text(csv_data, encoding="utf-8")

            X_train, X_test, y_train, y_test, feature_names = load_and_preprocess_tennis_data(
                file_path
            )

            nn_model = train_neural_network(X_train, y_train)
            model_path = Path(tmp_dir) / "model.pkl"
            save_model(nn_model, model_path)

            loaded_model = load_model(model_path)
            probas = predict_proba(loaded_model, X_test)

        self.assertEqual(len(probas), X_test.shape[0])
        for proba in probas:
            self.assertEqual(len(proba), 2)  # Binary classification
            self.assertAlmostEqual(sum(proba), 1.0, places=5)

    def test_save_load_json_results(self) -> None:
        """Test saving and loading JSON results."""
        results = {
            "models": {"decision_tree": {"accuracy": 0.85}},
            "feature_importance": {"decision_tree": [("Feature_1", 0.5)]},
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "results.json"
            save_model_json(results, json_path)

            with open(json_path, "r") as f:
                loaded = json.load(f)

        self.assertEqual(loaded["models"]["decision_tree"]["accuracy"], 0.85)

    def test_entrypoint_main_basic_mode(self) -> None:
        """Test CLI entrypoint in basic mode."""
        csv_data = """speed,spin,result
100,2000,1
110,2100,0
105,1900,1
"""

        with tempfile.TemporaryDirectory() as tmp_dir:
            csv_path = Path(tmp_dir) / "dataset.csv"
            model_path = Path(tmp_dir) / "model.json"
            csv_path.write_text(csv_data, encoding="utf-8")

            exit_code = entrypoint_main(
                [
                    "--data",
                    str(csv_path),
                    "--model-type",
                    "baseline",
                    "--output-dir",
                    str(tmp_dir) + "/",
                    "--feature-columns",
                    "speed",
                    "spin",
                    "--target-column",
                    "result",
                ]
            )

        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    import json
    unittest.main()


if __name__ == "__main__":
    unittest.main()
