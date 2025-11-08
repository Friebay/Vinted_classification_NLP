import os
import re
import spacy

import pandas as pd
from langdetect import detect

from multiprocessing import Pool

spacy_models = {
    'lt': spacy.load('lt_core_news_sm'),
    'en': spacy.load('en_core_web_sm'),
    'de': spacy.load('de_core_news_sm'),
}

multilingual = spacy.load('xx_ent_wiki_sm')


def pick_lang_model(text: str, spacy_models, multilingual):
    """ Pasirinkti kalbos modeli. """
    lang = detect(text)

    if lang in spacy_models:
        return spacy_models[lang]

    else:
        return multilingual


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

def parse_and_count_entities(data_row: dict, spacy_models, multilingual):
    # Prepare data
    text = data_row['item_description']

    text = remove_urls(text)
    text = remove_emails(text)

    spacy_model = pick_lang_model(text, spacy_models, multilingual)

    doc = spacy_model(text)
    data_row['ner_dict'] = get_ner_dict(doc)

    data_row['embedding'] = doc.vector

    return data_row


test_cases = [
    {'item_description': 'Toks nuo S iki L  nugaroje juodas suvarstomas kaspinas, galima plotį reguliuotis iki XL'},
    {'item_description': '41 dydis   Avalynė dėvėta, išvalius, nešiosote dar ilgai'},
    {'item_description': 'Nauji, šilko užvalkalai sandratiniai pagalvei   51×66    Vienato kaina 35€ '}
]

for case in test_cases:
    print(parse_and_count_entities(case, spacy_models, multilingual))