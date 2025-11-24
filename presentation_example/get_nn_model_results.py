from tensorflow.keras.models import load_model
import joblib
import numpy as np
import pandas as pd
import spacy
import re
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import os

model = load_model("best_nn_model.keras")
scaler = joblib.load("scaler.joblib")

classes = np.load("misc/classes.npy", allow_pickle=True)

spacy_models = {
    "lt": spacy.load("lt_core_news_lg"),
    "en": spacy.load("en_core_web_lg"),
    "de": spacy.load("de_core_news_lg"),
}

ALL_ENTITY_TYPES = [
    "CARDINAL", "DATE", "EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "MONEY",
    "NORP", "ORDINAL", "ORG", "PERCENT", "PERSON", "PRODUCT", "QUANTITY", "TIME",
    "WORK_OF_ART", "LOC", "MISC", "ORG", "PER",
]

def remove_urls(text):
    return re.sub(r"https?://\S+|www\.\S+", "", text)

def remove_emails(text):
    return re.sub(r"\S+@\S+", "", text)

def pick_lang_model(text, spacy_models):
    if not text or len(text) < 8:
        return None
    try:
        lang = detect(text)
    except (TypeError, LangDetectException):
        return None
    if lang in spacy_models:
        return spacy_models[lang]
    return None

def get_ner_dict(doc):
    output = {ent: 0 for ent in ALL_ENTITY_TYPES}
    output["numeric"] = 0
    output["currency"] = 0

    for ent in doc.ents:
        if ent.label_ in output:
            output[ent.label_] += 1 / len(doc)
    for tok in doc:
        if tok.is_digit:
            output["numeric"] += 1 / len(doc)
        if tok.is_currency:
            output["currency"] += 1 / len(doc)
    return output

def build_features(title, description):
    text = f"{title} {description}"
    text = remove_urls(text)
    text = remove_emails(text)

    spacy_model = pick_lang_model(text, spacy_models)
    if spacy_model is None:
        spacy_model = spacy_models.get("en")

    doc = spacy_model(text)
    embedding = doc.vector
    ner_dict = get_ner_dict(doc)

    df_template = pd.read_csv("final_df_2.csv", nrows=0)
    df_template = df_template.loc[:, ~df_template.columns.str.startswith("Unnamed: 0")]
    feature_cols = df_template.columns.tolist()[:-1]

    feature_row = {c: 0 for c in feature_cols}

    EMBED_SIZE = len(embedding)
    for i in range(EMBED_SIZE):
        col = f"embedding{i}"
        if col in feature_row:
            feature_row[col] = float(embedding[i])

    if "numeric" in feature_row:
        feature_row["numeric"] = float(ner_dict.get("numeric", 0))
    if "currency" in feature_row:
        feature_row["currency"] = float(ner_dict.get("currency", 0))

    for col in feature_cols:
        if col.startswith("embedding") or col in ("numeric", "currency"):
            continue
        if col in ner_dict:
            feature_row[col] = float(ner_dict.get(col, 0))

    row_df = pd.DataFrame([feature_row], columns=feature_cols)
    return row_df

def predict_text(title, description):
    X_df = build_features(title, description)
    X_scaled = scaler.transform(X_df)
    y_proba = model.predict(X_scaled, verbose=0)[0]
    y_pred = int(np.argmax(y_proba))
    pred_label = classes[y_pred] if classes is not None else y_pred
    
    return str(pred_label)

if __name__ == "__main__":
    title = """
    Suknelė
    """
    description = """
    Laisvo kritimo, lengva suknelė. Labiau L dydžiui
    
    """
    label = predict_text(title, description)
    print(f"Predicted class label: {label}")