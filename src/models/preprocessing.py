import pandas as pd
from sklearn.model_selection import train_test_split
import json
from ..config import DATASET_PATH, TRAIN_TEST_SPLIT_CONFIG, DATA_DIR


def load_and_preprocess_data():
    """Load and preprocess diabetes data."""
    df = pd.read_csv(DATASET_PATH)

    if 'PatientID' in df.columns:
        df = df.drop('PatientID', axis=1)

    feature_names = df.columns.tolist()[:-1]

    X = df.drop('Diabetic', axis=1).values
    y = df['Diabetic'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=TRAIN_TEST_SPLIT_CONFIG['test_size'],
                                                        random_state=TRAIN_TEST_SPLIT_CONFIG['random_state'],
                                                        stratify=y if TRAIN_TEST_SPLIT_CONFIG["stratify"] else None) # type: ignore

    split_metadata = {
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_positive_class": int(y_train.sum()),
        "test_positive_class": int(y_test.sum()),
        "feature_names": feature_names,
        "split_config": TRAIN_TEST_SPLIT_CONFIG
    }
    
    splits_dir = DATA_DIR / "splits"
    splits_dir.mkdir(exist_ok=True)

    with open(splits_dir / "split_metadata.json", "w") as f:
        json.dump(split_metadata, f, indent=2)
    
    return X_train, X_test, y_train, y_test, feature_names

def load_split_metadata():
    """Load the train/test split metadata."""
    with open(DATA_DIR / "splits" / "split_metadata.json", "r") as f:
        return json.load(f)
    

if __name__ == "__main__":
    pass