import spacy
import plac
from pathlib import Path
import xml.etree.cElementTree as ET
from sklearn.externals import joblib
from utils import extract_candidate_relations_from_sents ,get_dep_path ,extract_relation_features 

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
    for token in doc:
        print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
                token.shape_, token.is_alpha, token.is_stop)
    candidate_relations, candidate_labels = extract_candidate_relations_from_sents(
        doc, [])
    output = []
    clf, dv = joblib.load("model_svm_relations.pkl")

    for relation in candidate_relations:
        F = extract_relation_features(relation)
        feat_vec = dv.transform(F)
        general_type = clf.predict(feat_vec)[0]
        print(clf)
        print(general_type)
        if general_type != 'NONE':
            output.append((relation[0], relation[1],
                          relation[2], general_type))
    print(candidate_labels, candidate_relations)
    print(output)

if __name__ == "__main__":
    plac.call(main)
