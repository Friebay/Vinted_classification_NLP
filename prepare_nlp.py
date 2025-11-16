import os
import re

import numpy as np
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
import spacy

from sklearn.model_selection import StratifiedShuffleSplit
import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

spacy_models = {
    'lt': spacy.load('lt_core_news_lg'),
    'en': spacy.load('en_core_web_lg'),
    'de': spacy.load('de_core_news_lg'),
}

# https://github.com/explosion/spaCy/discussions/9147
ALL_ENTITY_TYPES = [
    "CARDINAL", 
    "DATE", 
    "EVENT", 
    "FAC", 
    "GPE", 
    "LANGUAGE", 
    "LAW", 
    "LOC", 
    "MONEY", 
    "NORP", 
    "ORDINAL", 
    "ORG", 
    "PERCENT", 
    "PERSON", 
    "PRODUCT", 
    "QUANTITY", 
    "TIME", 
    "WORK_OF_ART",
    "LOC", 
    "MISC", 
    "ORG", 
    "PER"
]


def pick_lang_model(text: str, spacy_models):
    """ Pasirinkti kalbos modeli. """

    if len(text) < 8:
        return None

    try:
        lang = detect(text)
    except (TypeError, LangDetectException):
        return None

    if lang in spacy_models:
        return spacy_models[lang]

    else:
        return None


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub('', text)


def remove_emails(text):
    email_pattern = r'\S+@\S+'
    return re.sub(email_pattern, '', text)


def get_ner_dict(doc):
    
    # Inicializuoti output dictionary su visais NER type'ais
    output = {
        ent: 0
        for ent in ALL_ENTITY_TYPES
    }

    output['numeric'] = 0
    output['currency'] = 0

    # spacy NER
    for ent in doc.ents:
        output[ent.label_] += 1 / len(doc)

    for tok in doc:
        # numerical
        if tok.is_digit:
            output['numeric'] += 1 / len(doc)

        # currency
        if tok.is_currency:
            output['currency'] += 1 / len(doc)

    return output


def parse_and_count_entities(data_row: dict):
    # Prepare data
    text = data_row['item_description']

    text = remove_urls(text)
    text = remove_emails(text)

    spacy_model = pick_lang_model(text, spacy_models)

    # Jei nepagavo vienos is top 3 kalbu
    if spacy_model is None:
        data_row['ner_dict'] = None
        data_row['embedding'] = None

        return data_row

    doc = spacy_model(text)
    data_row['ner_dict'] = get_ner_dict(doc)
    data_row['embedding'] = doc.vector

    return data_row


def tabular_data(res_df):
    """ Paversti netvarkingus, dict ir arrray tipo stulpelius i atskirus lenteles stulpelius """

    EMBED_SIZE = 300

    for i in range(EMBED_SIZE):
        res_df[f'embedding{i}'] = res_df['embedding'].apply(lambda x: x[i])

    res_df.drop(columns='embedding', inplace=True)
    df_ner = pd.json_normalize(res_df['ner_dict'])

    df_ner.index = res_df.index

    res_df = pd.concat([
        res_df.drop(columns=['ner_dict']),
        df_ner,
    ], axis='columns')

    return res_df


def encode_categories(res_df):
    cat1_encoder = LabelEncoder()
    cat1_encoder.fit( res_df['Sub_Category_1'] )

    np.save('classes.npy', cat1_encoder.classes_)

    return cat1_encoder.transform(res_df['Sub_Category_1'])


if __name__ == "__main__":

    df = pd.read_csv(
        "daiktai_translated.csv",
        sep="ʃ",
        engine="python",
        on_bad_lines='skip'
    )

    df_sample = df.groupby('Sub_Category_1', group_keys=False)\
        .apply(lambda x: x.sample(frac=0.05))

    print(df_sample.shape)

    df_records = df_sample.to_dict('records')

    del df

    # paralelizuoti parse_and_count_entities funkciją
    with Parallel(
        n_jobs=os.cpu_count() - 1,
        mmap_mode=None,
        backend="multiprocessing"
    ) as parallel:

        results = parallel(
            delayed(parse_and_count_entities)( data_row )
            for data_row in df_records
        )

    # surinkti rezultatus
    res_df = pd.DataFrame(results)
    res_df = res_df[['Sub_Category_1', 'ner_dict', 'embedding']]\
        .dropna()

    # dict ir array stulpelius "išpakuoti"
    res_df = tabular_data(res_df)

    # užkoduoti kategoriją
    category_encoded = encode_categories(res_df)
    res_df['cat'] = category_encoded

    res_df.drop(columns='Sub_Category_1', inplace=True)

    res_df.to_csv('final_df.csv')
