from collections import defaultdict, Counter
import numpy as np
import xgboost as xgb
import math
import spacy


def disambiguate_mentions(train_mentions, train_labels, dev_mentions, men_docs, parsed_entity_pages):
    docs_num = len(men_docs)
    men_num = len(train_mentions)
    tf_tokens = defaultdict(dict)
    tf_tokens = index_mension_doc(men_docs)
    label_list = []
    train_groups = []
    train_data = []
    for men_id, men_inf in train_mentions.items():
        train_groups.append(len(men_inf['candidate_entities']))
        for entities in men_inf['candidate_entities']:
            if entities == train_labels[men_id]['label']:
                label_list.append(1)
            else:
                label_list.append(0)
            feature1 = 0.0
            feature2 = 0.0
            feature3 = 0.0
            feature4 = 0
            feature5 = 0
            feature6 = 0.0
            feature7 = 0
            mention_upper_num = 0
            ent_upper_num = 0
            ent_len = len(entities)
            differ = ent_len - men_inf['length']

            # avoid overfitting
            for tuple in parsed_entity_pages[entities]:
                if men_inf['doc_title'] in tf_tokens[tuple[1]].keys():
                    num = tf_tokens[tuple[1]][men_inf['doc_title']]
                else:
                    num = 0
                feature7 += num
                if (tuple[3] != 'NOUN' and tuple[4] == 'O') or (tuple[3] == 'PROPN' and tuple[4] == 'B-GPE') or (
                        tuple[3] == 'PROPN' and tuple[4] == 'I-ORG'):
                    if men_inf['doc_title'] in tf_tokens[tuple[1]].keys():
                        tf_norm_num = 1.0 + math.log(1.0 + math.log(float(tf_tokens[tuple[1]][men_inf['doc_title']])))
                        idf_num = 1 + math.log(docs_num / 1.0 + len(tf_tokens[tuple[1]]))
                        tf_idf_num = tf_norm_num * idf_num
                    else:
                        tf_idf_num = 0.0
                    feature1 += tf_idf_num

            mention = men_inf['mention']
            mention_tokens = mention.split(" ")

            # the total number of tokens in mention
            len_mention_tokens = len(mention_tokens)

            # the sum of tf-idf values of all tokens in mention delimited by space
            for tok in mention_tokens:
                if men_inf['doc_title'] in tf_tokens[tok].keys():
                    tf_norm_num = 1.0 + math.log(1.0 + math.log(float(tf_tokens[tok][men_inf['doc_title']])))
                    idf_num = 1.0 + math.log(docs_num / (1.0 + len(tf_tokens[tok])))
                    tf_idf_num = tf_norm_num * idf_num
                else:
                    tf_idf_num = 0.0
                feature2 += tf_idf_num

                # the total number of capitalized tokens in mention
                if tok.isupper():
                    mention_upper_num += 1

            # the sum of tf-idf values of tokens within candidate_entities list
            entities_tokens = entities.split("_")
            len_entities_tokens = len(entities_tokens)

            # the sum of tf-idf values of tokens within candidate_entities list
            for ent in entities_tokens:
                if men_inf['doc_title'] in tf_tokens[ent].keys():
                    tf_norm_num = 1.0 + math.log(1.0 + math.log(float(tf_tokens[ent][men_inf['doc_title']])))
                    idf_num = 1.0 + math.log(docs_num / (1.0 + len(tf_tokens[ent])))
                    tf_idf_num = tf_norm_num * idf_num
                else:
                    tf_idf_num = 0.0
                feature3 += tf_idf_num

                # the total number of capitalized tokens in each entity within the candidate_entities list
                if ent.isupper():
                    ent_upper_num += 1

            # the total number of same tokens in entity and mentions with lowercase.
            feature4 = len([term for term in entities.lower().split('_') if term in mention.lower()])

            # the total number of same tokens in entity and mentions.
            for term in entities_tokens:
                if term in mention_tokens:
                    feature5 += 1

            # the rate between common length and candidate entities
            feature6 = feature4 / len_entities_tokens
            train_data.append([differ, feature1, feature2, feature3, feature4, feature5, feature6, len_entities_tokens,
                               len_mention_tokens, mention_upper_num, ent_upper_num, feature7])

    # dev_data construction
    dev_data = []
    dev_groups = []
    for men_id, men_inf in dev_mentions.items():
        dev_groups.append(len(men_inf['candidate_entities']))
        for entities in men_inf['candidate_entities']:
            feature1 = 0.0
            feature2 = 0.0
            feature3 = 0.0
            feature4 = 0
            feature5 = 0
            feature6 = 0.0
            feature7 = 0
            mention_upper_num = 0
            ent_upper_num = 0
            ent_len = len(entities)
            differ = ent_len - men_inf['length']
            
            # avoid overfitting
            for tuple in parsed_entity_pages[entities]:
                if men_inf['doc_title'] in tf_tokens[tuple[1]].keys():
                    num = tf_tokens[tuple[1]][men_inf['doc_title']]
                else:
                    num = 0
                feature7 += num
                if (tuple[3] != 'NOUN' and tuple[4] == 'O') or (tuple[3] == 'PROPN' and tuple[4] == 'B-GPE') or (
                        tuple[3] == 'PROPN' and tuple[4] == 'I-ORG'):
                    if men_inf['doc_title'] in tf_tokens[tuple[1]].keys():
                        tf_norm_num = 1.0 + math.log(1.0 + math.log(float(tf_tokens[tuple[1]][men_inf['doc_title']])))
                        idf_num = 1.0 + math.log(docs_num / 1.0 + len(tf_tokens[tuple[1]]))
                        tf_idf_num = tf_norm_num * idf_num
                    else:
                        tf_idf_num = 0.0
                    feature1 += tf_idf_num

            mention = men_inf['mention']
            mention_tokens = mention.split(" ")

            # the total number of tokens in mention
            len_mention_tokens = len(mention_tokens)

            # the sum of tf-idf values of all tokens in mention delimited by space
            for tok in mention_tokens:
                if men_inf['doc_title'] in tf_tokens[tok].keys():
                    tf_norm_num = 1.0 + math.log(1.0 + math.log(float(tf_tokens[tok][men_inf['doc_title']])))
                    idf_num = 1.0 + math.log(docs_num / (1.0 + len(tf_tokens[tok])))
                    tf_idf_num = tf_norm_num * idf_num
                else:
                    tf_idf_num = 0.0
                feature2 += tf_idf_num

                # the total number of capitalized tokens in mention
                if tok.isupper():
                    mention_upper_num += 1

            # the sum of tf-idf values of tokens within candidate_entities list
            entities_tokens = entities.split("_")
            len_entities_tokens = len(entities_tokens)

            # the sum of tf-idf values of tokens within candidate_entities list
            for ent in entities_tokens:
                if men_inf['doc_title'] in tf_tokens[ent].keys():
                    tf_norm_num = 1.0 + math.log(1.0 + math.log(float(tf_tokens[ent][men_inf['doc_title']])))
                    idf_num = 1 + math.log(docs_num / (1.0 + len(tf_tokens[ent])))
                    tf_idf_num = tf_norm_num * idf_num
                else:
                    tf_idf_num = 0.0
                feature3 += tf_idf_num
                
                # the total number of capitalized tokens in each entity within the candidate_entities list
                if ent.isupper():
                    ent_upper_num += 1

            # the total number of same tokens in entity and mentions with lowercase.
            feature4 = len([term for term in entities.lower().split('_') if term in mention.lower()])

            # the total number of same tokens in entity and mentions.
            for term in entities_tokens:
                if term in mention_tokens:
                    feature5 += 1

            # the rate between common length and candidate entities
            feature6 = feature4 / len_entities_tokens
            dev_data.append([differ, feature1, feature2, feature3, feature4, feature5, feature6, len_entities_tokens,
                             len_mention_tokens, mention_upper_num, ent_upper_num, feature7])

    train_labels = np.array(label_list)
    train_data = np.array(train_data)
    dev_data = np.array(dev_data)

    def transform_data(features, groups, labels=None):
        xgb_data = xgb.DMatrix(data=features, label=labels)
        xgb_data.set_group(groups)
        return xgb_data

    xgboost_train = transform_data(train_data, train_groups, train_labels)
    xgboost_test = transform_data(dev_data, dev_groups)
    # Parameters for XGBoost, you can fine-tune these parameters according to your settings...
    param = {'max_depth': 8, 'eta': 0.01, 'silent': 1, 'objective': 'rank:pairwise',
             'min_child_weight': 1, 'lambda': 100}
    # Train the classifier...
    classifier = xgb.train(param, xgboost_train, num_boost_round=4900)
    #  Predict test data...
    preds = classifier.predict(xgboost_test)
    idx = 0
    result = defaultdict(str)
    for iter_, group in enumerate(dev_groups):
        preds_list = preds[idx:idx + group].tolist()
        index_num = preds_list.index(max(preds_list))
        result[iter_ + 1] = dev_mentions[(iter_ + 1)]['candidate_entities'][index_num]
        idx += group
    return result


def index_mension_doc(men_docs):
    tf_entities = defaultdict(dict)
    tf_tokens = defaultdict(dict)
    # idf_entities = defaultdict(dict)
    # idf_tokens = defaultdict(dict)

    docs_num = len(men_docs)
    nlp = spacy.load("en_core_web_sm")
    for doc_title, doc_text in men_docs.items():
        doc = nlp(doc_text)
        for token in doc:
            if not token.is_stop and not token.is_punct:
                if token.text in tf_tokens.keys():
                    if doc_title not in tf_tokens[token.text].keys():
                        newdict = defaultdict(int)
                        newdict[doc_title] = 1
                        tf_tokens[token.text].update(newdict)
                    else:
                        tf_tokens[token.text][doc_title] += 1
                elif token.text not in tf_tokens.keys():
                    smalldict = defaultdict(int)
                    smalldict[doc_title] = 1
                    smalldict = dict(smalldict)
                    tf_tokens[token.text] = smalldict

        for ent in doc.ents:
            if ent.text in tf_entities.keys():
                if doc_title not in tf_entities[ent.text].keys():
                    newdict = defaultdict(int)
                    newdict[doc_title] = 1
                    tf_entities[ent.text].update(newdict)
                else:
                    tf_entities[ent.text][doc_title] += 1
            elif ent.text not in tf_entities.keys():
                small_dict = defaultdict(int)
                small_dict[doc_title] = 1
                small_dict = dict(small_dict)
                tf_entities[ent.text] = small_dict
            if ent.text in tf_tokens.keys():
                tf_tokens[ent.text][doc_title] -= 1
                if tf_tokens[ent.text][doc_title] == int(0):
                    del tf_tokens[ent.text][doc_title]
                if tf_tokens[ent.text] == {}:
                    del tf_tokens[ent.text]
    return tf_tokens
