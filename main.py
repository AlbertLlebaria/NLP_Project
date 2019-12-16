
from __future__ import unicode_literals, print_function
from pathlib import Path
from spacy.util import minibatch, compounding
from sklearn.externals import joblib
from utils import parse_XML_dataset, test_NER, test_PARSE
import plac
import random
import spacy


# new entity label
NER_LABELS = ['TRAJECTOR', 'SPATIAL_INDICATOR', 'LANDMARK']
PARSE_LABELS = ['TRAJECTOR', '-', 'LANDMARK', 'SPATIAL_INDICATOR']

NER_TRAIN_DATA, PARSE_TRAIN_DATA, RELATIONS_TRAIN = parse_XML_dataset(
    './dataset_1/train.xml')
NER_TEST_DATA, PARSE_TEST_DATA, RELATIONS_TEST = parse_XML_dataset(
    './dataset_1/test.xml')


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="SpRL", output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")

    # Add entity recognizer to model if it's not in the pipeline
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    else:
        ner = nlp.get_pipe("ner")
    for LABEL in NER_LABELS:
        ner.add_label(LABEL)  # add new entity label to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()

    if "parser" in nlp.pipe_names:
        nlp.remove_pipe("parser")
    parser = nlp.create_pipe("parser")
    nlp.add_pipe(parser, first=True)

    for label in PARSE_LABELS:
        parser.add_label(label)

    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "parser"]
    with nlp.disable_pipes(*other_pipes):  # only train parser
        optimizer = nlp.begin_training()
        for itn in range(n_iter):
            random.shuffle(PARSE_TRAIN_DATA)
            losses = {}
            # batch up the examples using spaCy's minibatch
            batches = minibatch(
                PARSE_TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, losses=losses)
            print("Losses", losses)

    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(NER_TRAIN_DATA)
            batches = minibatch(NER_TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer,
                           drop=0.35, losses=losses)
            print("Losses", losses)
    aux = 'one child is carrying shoe-cleaning equipment , the one on the left is holding a cup , the other one is hanging on to the fence .'
    # test the trained model
    print("+=+=+=+=+=+=+=+=+=+=+ TESTING NER +=+=+=+=+=+=+=+=+=+=+")
    test_NER(nlp, NER_TEST_DATA)
    print("+=+=+=+=+=+=+=+=+=+=+ TESTING PARSE +=+=+=+=+=+=+=+=+=+=+")
    test_PARSE(nlp, PARSE_TEST_DATA, RELATIONS_TEST)

    # save model to output directory
    if output_dir is not None:
        output_dir = Path(output_dir)
        if not output_dir.exists():
            output_dir.mkdir()
        nlp.meta["name"] = new_model_name  # rename model
        nlp.to_disk(output_dir)
        print("Saved model to", output_dir)

        # test the saved model
        print("Loading from", output_dir)
        nlp2 = spacy.load(output_dir)
        # Check the classes have loaded back consistently
        assert nlp2.get_pipe("ner").move_names == move_names
        doc2 = nlp2(aux)
        print(doc2)
        for ent in doc2.ents:
            print(ent.label_, ent.text)


if __name__ == "__main__":
    plac.call(main)
