# src/metadata/dataset.py

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def load_metadata_dataframe(input_path: str | Path) -> pd.DataFrame:
    """
    Đọc file metadata gốc và giữ lại các cột cần thiết cho metadata model.
    Không làm EDA, không split – chỉ load + filter cột.
    """
    input_path = Path(input_path)
    df_raw = pd.read_csv(input_path)

    keep_cols = [
        "tconst",
        "titleType",
        "startYear",
        "runtimeMinutes",
        "genres",
        "averageRating",
        "sex_code",
        "violence_code",
        "profanity_code",
        "drug_code",
        "intense_code",
        "final_certificate",  # label
    ]

    missing = [c for c in keep_cols if c not in df_raw.columns]
    if missing:
        raise ValueError(f"Các cột thiếu trong file input: {missing}")

    df = df_raw[keep_cols].copy()
    return df


def preprocess_and_split_metadata(
    df: pd.DataFrame,
    label_col: str = "final_certificate",
    id_col: str = "tconst",
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
):
    """
    - Encode genres (multi-hot)
    - One-hot titleType
    - Tách X, y
    - Train/val/test split (stratified)
    - Tính sample_weight_train dựa trên tần suất label ở train
    """
    df = df.copy()

    # 1. Encode genres (multi-hot)
    if "genres" not in df.columns:
        raise ValueError("Cột 'genres' không tồn tại trong dataframe.")

    genres_clean = df["genres"].fillna("").str.replace(" ", "", regex=False)
    genres_dummies = genres_clean.str.get_dummies(sep=",")
    df = pd.concat([df.drop(columns=["genres"]), genres_dummies], axis=1)

    # 2. One-hot titleType (nếu cột tồn tại)
    if "titleType" in df.columns:
        title_dummies = pd.get_dummies(df["titleType"], prefix="titleType")
        df = pd.concat([df.drop(columns=["titleType"]), title_dummies], axis=1)

    # 3. Chuẩn bị X, y
    if label_col not in df.columns:
        raise ValueError(f"Cột label '{label_col}' không tồn tại trong dataframe.")
    if id_col not in df.columns:
        raise ValueError(f"Cột id '{id_col}' không tồn tại trong dataframe.")

    y_str = df[label_col].copy()
    X = df.drop(columns=[label_col, id_col])

    le = LabelEncoder()
    y = le.fit_transform(y_str)
    num_classes = len(le.classes_)
    if num_classes < 2:
        raise ValueError("Số lượng lớp < 2, kiểm tra lại cột final_certificate.")

    # 4. Train / Val / Test split
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # val_size là tỉ lệ trên full data -> quy đổi sang tỉ lệ trên X_temp
    val_ratio_on_temp = val_size / (1.0 - test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_on_temp,
        random_state=random_state,
        stratify=y_temp,
    )

    # 5. Class weights & sample_weight cho train
    label_counts = pd.Series(y_train).value_counts()
    max_count = label_counts.max()
    class_weights = {cls: max_count / cnt for cls, cnt in label_counts.items()}
    sample_weight_train = np.array([class_weights[c] for c in y_train])

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        le,
        sample_weight_train,
    )


if __name__ == "__main__":
    data_path = Path("data") / "metadata_input.csv"
    df_meta = load_metadata_dataframe(data_path)

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        le,
        sample_weight_train,
    ) = preprocess_and_split_metadata(df_meta)

    print("Train shape:", X_train.shape)
    print("Val shape  :", X_val.shape)
    print("Test shape :", X_test.shape)
    print("Classes    :", list(le.classes_))
