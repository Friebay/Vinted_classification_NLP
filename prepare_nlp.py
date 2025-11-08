import os
import re

from joblib import Parallel, delayed
import spacy

import pandas as pd
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

spacy_models = {
    'lt': spacy.load('lt_core_news_sm'),
    'en': spacy.load('en_core_web_sm'),
    'de': spacy.load('de_core_news_sm'),
}


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
    output = {
        'numeric': 0,
        'currency': 0
    }

    # spacy NER
    for ent in doc.ents:
        if ent.label_ not in output:
            output[ent.label_] = 1

        else:
            output[ent.label_] += 1

    for tok in doc:
        # numerical
        if tok.is_digit:
            output['numeric'] += 1

        # currency
        if tok.is_currency:
            output['currency'] += tok.is_currency

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

df = pd.read_csv(
    "daiktai_cleaned.csv",
    sep="ʃ",
    engine="python",
    on_bad_lines='skip'
)
df_records = df.to_dict('records')

with Parallel(
    n_jobs=os.cpu_count() - 1,
    mmap_mode=None,
    backend="multiprocessing"
) as parallel:

    results = parallel(
        delayed(parse_and_count_entities)( data_row )
        for data_row in df_records
    )
print('Done')

res_df = pd.DataFrame(results)
res_df.dropna().to_csv('prepared_df.csv')
print(res_df.head())