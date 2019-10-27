import spacy
from sprl import *

nlp = spacy.load('models/en_core_web_lg-sprl')

sentence = "An angry big dog is behind us."

rel = sprl(sentence, nlp, model_relext_filename='models/model_svm_relations.pkl')

print(rel)