#!/usr/bin/env python
# coding: utf8
"""Example of training an additional entity type

This script shows how to add a new entity type to an existing pretrained NER
model. To keep the example short and simple, only four sentences are provided
as examples. In practice, you'll need many more — a few hundred would be a
good start. You will also likely need to mix in examples of other entity
types, which might be obtained by running the entity recognizer over unlabelled
sentences, and adding their annotations to the training set.

The actual training is performed by looping over the examples, and calling
`nlp.entity.update()`. The `update()` method steps through the words of the
input. At each word, it makes a prediction. It then consults the annotations
provided on the GoldParse instance, to see whether it was right. If it was
wrong, it adjusts its weights so that the correct action will score higher
next time.

After training your model, you can save it to a directory. We recommend
wrapping models as Python packages, for ease of deployment.

For more details, see the documentation:
* Training: https://spacy.io/usage/training
* NER: https://spacy.io/usage/linguistic-features#named-entities

Compatible with: spaCy v2.1.0+
Last tested with: v2.1.0
"""
from __future__ import unicode_literals, print_function

import plac
import random
from pathlib import Path
import spacy
from spacy.util import minibatch, compounding
import xml.etree.ElementTree as ET
import toolz

# new entity label
LABELS = ['TRAJECTOR', 'SPATIAL_INDICATOR', 'LANDMARK']
root_test = ET.parse('./test.xml').getroot()
root_train = ET.parse('./train.xml').getroot()

TRAIN_DATA = []
TEST_DATA = []


def find_word_indexes(word, text_array):
    found = False
    index = 0
    word_index = 0

    while(found is False and index < len(text_array)-1):
        if text_array[index].lower() == word.lower():
            found = True
        else:
            word_index += len(text_array[index]) + 1
            index += 1
    if(index == len(text_array)-1):
        return word_index - 1
    else:
        return word_index


for sentence in root_train:
    text = sentence.find('CONTENT').text.lstrip()
    TRAJECTORS_xml = sentence.findall('TRAJECTOR')
    SPATIALS_I_xml = sentence.findall('SPATIAL_INDICATOR')
    LANDMARKS_indicator_xml = sentence.findall('LANDMARK')
    entities = []
    word_list = text.split()

    for TRAJECTOR in TRAJECTORS_xml:
        word = TRAJECTOR.text.lower().split()[0]
        pos = find_word_indexes(word, word_list)
        entities.append((pos,
                             pos + len(word), 'TRAJECTOR'))

    for SPATIAL in SPATIALS_I_xml:
        word = SPATIAL.text.lower().split()[0]
        pos = find_word_indexes(word, word_list)
        entities.append((pos,
                             pos + len(word), 'SPATIAL_INDICATOR'))

    for LANDMARK in LANDMARKS_indicator_xml:
        word = LANDMARK.text.lower().split()[0]
        pos = find_word_indexes(word, word_list)
        entities.append((pos,
                             pos + len(word), 'LANDMARK'))

    TRAIN_DATA.append(
        (text, {"entities": list(toolz.unique(entities, key=lambda x: x[0]))}))


for sentence in root_test:
    text = sentence.find('CONTENT').text.lstrip()
    TRAJECTORS_xml = sentence.findall('TRAJECTOR')
    SPATIALS_I_xml = sentence.findall('SPATIAL_INDICATOR')
    LANDMARKS_indicator_xml = sentence.findall('LANDMARK')
    entities = []
    word_list = text.split()

    for TRAJECTOR in TRAJECTORS_xml:
        word = TRAJECTOR.text.lower().split()[0]
        pos = find_word_indexes(word, word_list)
        entities.append((pos,
                             pos + len(word), 'TRAJECTOR'))

    for SPATIAL in SPATIALS_I_xml:
        word = SPATIAL.text.lower().split()[0]
        pos = find_word_indexes(word, word_list)
        entities.append((pos,
                             pos + len(word), 'SPATIAL_INDICATOR'))

    for LANDMARK in LANDMARKS_indicator_xml:
        word = LANDMARK.text.lower().split()[0]
        pos = find_word_indexes(word, word_list)
        entities.append((pos,
                             pos + len(word), 'LANDMARK'))

    TEST_DATA.append(
        (text, {"entities": list(toolz.unique(entities, key=lambda x: x[0]))}))


@plac.annotations(
    model=("Model name. Defaults to blank 'en' model.", "option", "m", str),
    new_model_name=("New model name for model meta.", "option", "nm", str),
    output_dir=("Optional output directory", "option", "o", Path),
    n_iter=("Number of training iterations", "option", "n", int),
)
def main(model=None, new_model_name="animal", output_dir=None, n_iter=30):
    """Set up the pipeline and entity recognizer, and train the new entity."""
    random.seed(0)
    if model is not None:
        nlp = spacy.load(model)  # load existing spaCy model
        print("Loaded model '%s'" % model)
    else:
        nlp = spacy.blank("en")  # create blank Language class
        print("Created blank 'en' model")
    # Add entity recognizer to model if it's not in the pipeline
    # nlp.create_pipe works for built-ins that are registered with spaCy
    if "ner" not in nlp.pipe_names:
        ner = nlp.create_pipe("ner")
        nlp.add_pipe(ner)
    # otherwise, get it, so we can add labels to it
    else:
        ner = nlp.get_pipe("ner")
    for LABEL in LABELS:
        ner.add_label(LABEL)  # add new entity label to entity recognizer

    if model is None:
        optimizer = nlp.begin_training()
    else:
        optimizer = nlp.resume_training()
    move_names = list(ner.move_names)
    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):  # only train NER
        sizes = compounding(1.0, 4.0, 1.001)
        # batch up the examples using spaCy's minibatch
        for itn in range(n_iter):
            random.shuffle(TRAIN_DATA)
            batches = minibatch(TRAIN_DATA, size=sizes)
            losses = {}
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer,
                           drop=0.35, losses=losses)
            print("Losses", losses)
    aux = 'one child is carrying shoe-cleaning equipment , the one on the left is holding a cup , the other one is hanging on to the fence .'
    # test the trained model
    for test_text in TEST_DATA:
        doc = nlp(test_text[0])
        print("Entities in '%s'" % test_text[0])
        for ent in doc.ents:
            print('PREDICTED : ', ent.label_, ent.text)
            print('ACTUAL :', test_text[1])

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
