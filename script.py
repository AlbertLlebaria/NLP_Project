import spacy
import plac
from pathlib import Path
import xml.etree.cElementTree as ET
from sklearn.externals import joblib
import utils
import numpy as np
labels = ['LANDMARK', 'TRAJECTOR', 'SPATIAL_INDICATOR']


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    sentence=("Sentence to apply the spatial labeling.", "option", "s", str),
    output_dir=("Optional output directory", "option", "o", Path),
)
def main(model=None, sentence="", output_dir=None):
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
    else:
        print("Model not found")
    print("Loaded model '%s'" % model)
    print("Loaded sentence '%s'" % sentence)

    doc = nlp(sentence)
    print(f"In sentence {sentence} the model has found the following spatial entities:")
    for entity in doc.ents:
        print(f"{entity.text} as  {entity.label_}")
    candidate_relations, candidate_labels = utils.extract_candidate_relations_from_sents(
        doc, [])
    output = []
    clf, dv = joblib.load("model_svm_relations.pkl")
    print(f"In sentence {sentence} the model has found the following spatial relations:")
    for relation in candidate_relations:
            F = utils.extract_relation_features(relation)
            feat_vec = dv.transform(F)
            general_type = clf.predict(feat_vec)[0]

            if general_type != 'NONE':
                output.append((relation[0], relation[1],
                               relation[2], general_type))

    print(output)
if __name__ == "__main__":
    plac.call(main)
