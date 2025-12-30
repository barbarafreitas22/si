from unittest import TestCase
import os
from datasets import DATASETS_PATH

from si.io.csv_file import read_csv
from si.model_selection.split import train_test_split
from si.models.knn_classifier import KNNClassifier
from si.models.logistic_regression import LogisticRegression
from si.models.decision_tree_classifier import DecisionTreeClassifier
from si.ensemble.stacking_classifier import StackingClassifier

class TestStackingClassifier(TestCase):

    def test_protocol_exercise_10(self):
        csv_file = os.path.join(DATASETS_PATH, 'breast_bin', 'breast-bin.csv')
        
        dataset = read_csv(filename=csv_file, label=True, sep=",")

        # Split the data into train and test sets 
        train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=42)

        # Create a KNNClassifier model 
        knn = KNNClassifier(k=3)

        # Create a LogisticRegression model 
        log_reg = LogisticRegression(l2_penalty=0.1, alpha=0.001, max_iter=1000)

        # Create a DecisionTree model 
        dt = DecisionTreeClassifier(min_sample_split=2, max_depth=5)

        # Create a second KNNClassifier model 
        final_knn = KNNClassifier(k=3)

        # Create a StackingClassifier model using the previous classifiers 
        stacking = StackingClassifier(
            models=[knn, log_reg, dt],
            final_model=final_knn
        )

        #  Train the StackingClassifier model
        stacking.fit(train_dataset)

        score = stacking.score(test_dataset)
        
        print(f"\n------------------------------------------------")
        print(f"8. Score: {score:.4f}")
        print(f"------------------------------------------------")

        # Extra asserts to validate the score is within expected range
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        self.assertGreater(score, 0.90)