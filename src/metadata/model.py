# src/metadata/model.py

from pathlib import Path

from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix


def train_xgb_metadata(
    X_train,
    y_train,
    X_val,
    y_val,
    sample_weight_train,
    num_classes: int,
    verbose: int = 50,
):
    """
    Train XGBoost cho metadata branch.

    Trả về:
        xgb_clf: model đã train
    """
    xgb_clf = XGBClassifier(
        objective="multi:softprob",
        num_class=num_classes,
        eval_metric="mlogloss",
        learning_rate=0.05,
        max_depth=6,
        n_estimators=400,
        subsample=0.8,
        colsample_bytree=0.8,
        tree_method="hist",
        random_state=42,
    )

    print("\n>>> Training XGBoost (metadata)...")
    xgb_clf.fit(
        X_train,
        y_train,
        sample_weight=sample_weight_train,
        eval_set=[(X_val, y_val)],
        verbose=verbose,
    )

    return xgb_clf


import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_xgb_metadata(model, X_test, y_test, label_encoder):
    y_pred = model.predict(X_test)

    # Lấy đúng các label thực sự xuất hiện trong y_test hoặc y_pred
    labels = np.unique(np.concatenate([y_test, y_pred]))
    class_names = label_encoder.inverse_transform(labels)

    print("\n=== XGBoost (metadata) classification report ===")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=labels,
            target_names=class_names,
        )
    )

    cm = confusion_matrix(y_test, y_pred, labels=labels)
    print("XGBoost confusion matrix:")
    print(cm)


def save_xgb_model(model, output_path: str | Path):
    """
    Lưu XGBoost model ra file .json
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model.save_model(str(output_path))
    print(f"\nSaved XGBoost model to {output_path}")


