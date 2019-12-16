import xml.etree.ElementTree as ET
import spacy
import numpy as np
from sklearn.externals import joblib


def parse_XML_dataset(path):
    en_core = spacy.load('en_core_web_lg')

    data = ET.parse(path).getroot()
    NER = []
    PARSER = []
    S_RELATIONS = []
    for sentence in data:
        text = sentence.find('CONTENT').text.lstrip()
        TRAJECTORS_xml = sentence.findall('TRAJECTOR')
        SPATIALS_I_xml = sentence.findall('SPATIAL_INDICATOR')
        LANDMARKS_indicator_xml = sentence.findall('LANDMARK')
        RELATIONS_xml = sentence.findall('RELATION')

        # ENTITY RECOGNITION DATA
        entities = []
        positions = []

        doc = en_core(text)

        for TRAJECTOR in TRAJECTORS_xml:
            id = int(TRAJECTOR.attrib['id'][2:len(TRAJECTOR.attrib['id'])])
            if(id < len(doc) and doc[id].idx not in positions):
                entities.append((doc[id].idx,
                                 doc[id].idx + len(doc[id].text), 'TRAJECTOR'))
                positions.append(doc[id].idx)

        for SPATIAL in SPATIALS_I_xml:
            id = int(SPATIAL.attrib['id'][2:len(SPATIAL.attrib['id'])])
            if(id < len(doc) and doc[id].idx not in positions):
                entities.append((doc[id].idx,
                                 doc[id].idx + len(doc[id].text), 'SPATIAL_INDICATOR'))
                positions.append(doc[id].idx)

        for LANDMARK in LANDMARKS_indicator_xml:
            id = int(LANDMARK.attrib['id'][2:len(LANDMARK.attrib['id'])])
            if(id < len(doc) and doc[id].idx not in positions):
                entities.append((doc[id].idx,
                                 doc[id].idx + len(doc[id].text), 'LANDMARK'))
                positions.append(doc[id].idx)

        NER.append(
            (text, {"entities": entities}))

        # PARSE DATA
        heads = [token.head.i for token in doc]
        s_relations = []
        deps = ['-' for token in doc]
        for relation in RELATIONS_xml:
            trajector_id = int(
                relation.attrib['tr'][2:len(relation.attrib['tr'])])
            landmark_id = int(
                relation.attrib['lm'][2:len(relation.attrib['lm'])])
            spatial_ind_id = int(
                relation.attrib['sp'][2:len(relation.attrib['sp'])])

            if(trajector_id < len(doc)):
                deps[trajector_id] = 'TRAJECTOR'
            if(landmark_id < len(doc)):
                deps[landmark_id] = 'LANDMARK'
            if(spatial_ind_id < len(doc)):
                deps[spatial_ind_id] = 'SPATIAL_INDICATOR'
            s_relations.append(dict(
                LANDMARK=landmark_id,
                TRAJECTOR=trajector_id,
                SPATIAL_INDICATOR=spatial_ind_id))

        S_RELATIONS.append(s_relations)
        PARSER.append((text, {
            'deps': deps,
            'heads': heads
        }))

    return NER, PARSER, S_RELATIONS


def find_trueP_and_falseN(found_entities, true_entity):
    isPositive = False
    for ent in found_entities:
        if(ent.end_char == true_entity[1] and ent.start_char == true_entity[0] and ent.label_ == true_entity[2]):
            isPositive = True
    return 1 if isPositive else 0


def find_falseP(true_entities, entity_found):
    found = False
    for ent in true_entities:
        if(entity_found.end_char == ent[1] and entity_found.start_char == ent[0] and entity_found.label_ == ent[2]):
            found = True
    return 1 if found else 0

# tp is the number of true positives (the number of instances that are correctly found),
# fp is the number of false positives (number of instances that are predicted by the system but not a true instance)


def precision(tp, fp):
    try:
        return tp/(tp+fp)
    except:
        return 0

# tp is the number of true positives (the number of instances that are correctly found),
# fn is the number of false negatives (missing results).


def recall(tp, fn):
    try:
        return tp/(tp+fn)
    except:
        return 0


def F1_score(tp, fn, fp):
    prec = precision(tp, fp)
    rec = recall(tp, fn)
    try:
        f1 = 2 * ((prec*rec)/(prec+rec))
        return prec, rec, f1
    except:
        return prec, rec, 0


def test_NER(nlp, NER_TEST):
    f1_avg = []
    prec_avg = []
    rec_avg = []

    f1_avg_role = dict(LANDMARK=[], TRAJECTOR=[], SPATIAL_INDICATOR=[])
    prec_avg_role = dict(LANDMARK=[], TRAJECTOR=[], SPATIAL_INDICATOR=[])
    rec_avg_role = dict(LANDMARK=[], TRAJECTOR=[], SPATIAL_INDICATOR=[])

    for test_text in NER_TEST:
        doc = nlp(test_text[0])
        entities_found = doc.ents
        detailed = dict(LANDMARK=dict(tp=0, fp=0, fn=0), TRAJECTOR=dict(
            tp=0, fp=0, fn=0), SPATIAL_INDICATOR=dict(tp=0, fp=0, fn=0))
        for true_entity in test_text[1]['entities']:
            isTruePositive = False
            for found in entities_found:
                if(found.label_ == true_entity[2] and found.end_char == true_entity[1] and found.start_char == true_entity[0]):
                    isTruePositive = True
            if(isTruePositive):
                detailed[true_entity[2]]['tp'] += 1
            else:
                detailed[true_entity[2]]['fn'] += 1
        for found in entities_found:
            isFp = find_falseP(test_text[1]['entities'], found)
            if(isFp == 0):
                detailed[found.label_]['fp'] += 1

        for k in detailed.keys():
            tp_d = detailed[k]['tp']
            fp_d = detailed[k]['fp']
            fn_d = detailed[k]['fn']
            prec, rec, f1 = F1_score(tp_d, fn_d, fp_d)

            if(tp_d == 0 and fp_d == 0 and fn_d == 0):
                prec, rec, f1 = 1, 1, 1

            f1_avg_role[k].append(f1)
            prec_avg_role[k].append(prec)
            rec_avg_role[k].append(rec)

        tp = sum([val['tp'] for val in detailed.values()])
        fn = sum([val['fn'] for val in detailed.values()])
        fp = sum([val['fp'] for val in detailed.values()])
        if(len(entities_found) == 0 and len(test_text[1]['entities']) == 0):
            print(f'F1: 1, no entities found when no entities where suposed to be found')
            f1_avg.append(1)
            prec_avg.append(1)
            rec_avg.append(1)
        else:
            prec, rec, f1 = F1_score(tp, fn, fp)
            print(f'F1: {f1} , precision: {prec}, Recall: {rec}')
            f1_avg.append(f1)
            prec_avg.append(prec)
            rec_avg.append(rec)

    for k in detailed.keys():
        print(
            f"{k} : {np.mean(f1_avg_role[k])},{np.mean(prec_avg_role[k])}, {np.mean(rec_avg_role[k])}")
    print(f"MEAN: {np.mean(f1_avg)},{np.mean(prec_avg)}, {np.mean(rec_avg)}")


def get_dep_path(span1, span2):
    assert span1.sent == span2.sent, "sent1: {}, span1: {}, sent2: {}, span2: {}".format(
        span1.sent, span1, span2.sent, span2)

    up = []
    down = []

    head = span1[0]
    while head.dep_ != 'ROOT':
        up.append(head)
        head = head.head
    up.append(head)

    head = span2[0]
    while head.dep_ != 'ROOT':
        down.append(head)
        head = head.head
    down.append(head)
    down.reverse()

    for n1, t1 in enumerate(up):
        for n2, t2 in enumerate(down):
            if t1 == t2:
                return ["{}::{}".format(u.dep_, 'up') for u in up[1:n1]] + ["{}::{}".format(d.dep_, 'down') for d in down[n2:]]


def extract_relation_features(relation):
    F = {}  # Feature dict

    trigger = relation[1]
    args = [relation[0], relation[2]]

    # Extract features relating to trigger
    #trigger_head = get_head(trigger)

    for n, token in enumerate(trigger):
        F['TF1T{}'.format(n)] = token.text
        F['TF2T{}'.format(n)] = token.lemma_
        F['TF3T{}'.format(n)] = token.pos_
        F['TF4T{}'.format(n)] = "::".join(
            [token.lemma_, token.pos_])  # RF.2 concat RF.1

    # Extract features relating to the two arguments
    for a, arg in enumerate(args):
        if arg is not None:
            for n, token in enumerate(arg):
                F['A{}F5T{}'.format(a, n)] = token.text
                F['A{}F6T{}'.format(a, n)] = token.lemma_
                F['A{}F7T{}'.format(a, n)] = token.pos_
                F['A{}F8T{}'.format(a, n)] = "::".join(
                    [token.lemma_, token.pos_])

            if arg[-1].i < trigger[0].i:
                F['A{}F12'.format(a)] = 'LEFT'
                F['A{}F22'.format(a)] = trigger[0].i - arg[-1].i
            elif arg[0].i > trigger[-1].i:
                F['A{}F12'.format(a)] = 'RIGHT'
                F['A{}F22'.format(a)] = arg[0].i - trigger[-1].i

            path = get_dep_path(arg, trigger)
            for np, p in enumerate(path):
                F['A{}F17E{}'.format(a, np)] = p
            F['A{}F20'.format(a)] = len(path)
            F['A{}F24'.format(a)] = False
        else:
            F['A{}F24'.format(a)] = True

    # Joint features
    if 'A0F12' in F and 'A1F12' in F:
        F['F13'] = "::".join([F['A0F12'], F['A1F12']])
        if F['A0F12'] == F['A1F12']:
            F['14'] = True
        else:
            F['14'] = False

    if 'F13' in F:
        for n, token in enumerate(trigger):
            F['14T{}'.format(n)] = '::'.join([F['F13'], token.lemma_])

    if 'A0F22' in F and 'A1F22' in F:
        F['F23'] = F['A0F22'] + F['A1F22']

    return F


def extract_candidate_relations_from_sents(doc, gold_relations):
    candidate_relations = []
    candidate_labels = []

    triggers = [t for t in doc.ents if t.label_ == 'SPATIAL_INDICATOR']
    trajectors = [t for t in doc.ents if t.label_ == 'TRAJECTOR']
    landmarks = [t for t in doc.ents if t.label_ == 'LANDMARK']

    for trigger in triggers:
        for trajector in trajectors:
            for landmark in landmarks:
                if not (trajector is None and landmark is None):
                    try:
                        assert trajector.sent == trigger.sent == landmark.sent, "{}: {}".format(
                            doc, doc.ents)
                        crel = (trajector, trigger, landmark)
                        if crel not in gold_relations:
                            candidate_relations.append(crel)
                            candidate_labels.append('NONE')
                        else:
                            pass
                    except:
                        pass
    return candidate_relations, candidate_labels


def test_RELATIONS(nlp, PARSE_TEST, s_relations):
    f1_avg = []
    prec_avg = []
    rec_avg = []

    for sentence_index, test_text in enumerate(PARSE_TEST):
        doc = nlp(test_text[0])

        candidate_relations, candidate_labels = extract_candidate_relations_from_sents(
            doc, [])
        output = []
        clf, dv = joblib.load("model_svm_relations.pkl")

        for relation in candidate_relations:
            F = extract_relation_features(relation)
            feat_vec = dv.transform(F)
            general_type = clf.predict(feat_vec)[0]

            if general_type != 'NONE':
                output.append((relation[0], relation[1],
                               relation[2], general_type))
        tp = 0
        fp = 0
        fn = 0

        for relation in s_relations[sentence_index]:
            found = True
            # prediction is a tuple of (trajector, spatial_indicator, landmark)
            for predicted in output:
                # print([el.start for el in predicted[0:3]], relation,test_text[0])
                if(predicted[0].start != relation['TRAJECTOR']):
                    found = False
                elif (predicted[1].start != relation['SPATIAL_INDICATOR']):
                    found = False
                elif(predicted[2].start != relation['LANDMARK']):
                    found = False

                if(found):
                    tp += 1
                else:
                    fn += 1
        diff = len(output) - len(s_relations[sentence_index])
        fn = diff if diff >= 0 else 0
        if(len(output) == 0 and len(s_relations[sentence_index]) == 0):
            print(f'F1: 1, precisio: 1, recall : 1 ')
            f1_avg.append(1)
            prec_avg.append(1)
            rec_avg.append(1)
        else:
            prec, rec, f1 = F1_score(tp, fn, fp)
            print(f'F1: {f1} , precision: {prec}, Recall: {rec}')
            f1_avg.append(f1)
            prec_avg.append(prec)
            rec_avg.append(rec)
    print(np.mean(f1_avg), np.mean(prec_avg), np.mean(rec_avg))
