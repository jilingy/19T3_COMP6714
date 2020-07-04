## Import Libraries and Modules here...
import spacy
import math
import copy
from itertools import combinations
from collections import defaultdict, Counter


class InvertedIndex:
    def __init__(self):
        self.tf_tokens = defaultdict(dict)
        self.tf_entities = defaultdict(dict)
        self.idf_tokens = defaultdict(dict)
        self.idf_entities = defaultdict(dict)
        self.nlp = spacy.load("en_core_web_sm")
        self.docs_num = int

    def index_documents(self, documents):
        self.docs_num = len(documents)
        for key, value in documents.items():
            doc = self.nlp(value)
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    if token.text in self.tf_tokens.keys():
                        if key not in self.tf_tokens[token.text].keys():
                            newdict = defaultdict(int)
                            newdict[key] = 1
                            self.tf_tokens[token.text].update(newdict)
                        else:
                            self.tf_tokens[token.text][key] += 1
                    elif token.text not in self.tf_tokens.keys():
                        smalldict = defaultdict(int)
                        smalldict[key] = 1
                        smalldict = dict(smalldict)
                        self.tf_tokens[token.text] = smalldict

            for ent in doc.ents:
                if ent.text in self.tf_entities.keys():
                    if key not in self.tf_entities[ent.text].keys():
                        newdict = defaultdict(int)
                        newdict[key] = 1
                        self.tf_entities[ent.text].update(newdict)
                    else:
                        self.tf_entities[ent.text][key] += 1
                elif ent.text not in self.tf_entities.keys():
                    small_dict = defaultdict(int)
                    small_dict[key] = 1
                    small_dict = dict(small_dict)
                    self.tf_entities[ent.text] = small_dict
                if ent.text in self.tf_tokens.keys():
                    self.tf_tokens[ent.text][key] -= 1
                    if self.tf_tokens[ent.text][key] == int(0):
                        del self.tf_tokens[ent.text][key]
                    if self.tf_tokens[ent.text] == {}:
                        del self.tf_tokens[ent.text]

    def split_query(self, Q, DoE):
        query_list = Q.split(" ")
        selected_list = []
        for i in range(1, len(query_list) + 1):
            comb_list = list(combinations(query_list, i))
            for item in comb_list:
                word = " ".join(item)
                if word in DoE.keys():
                    selected_list.append(word)
        selected_list = list(set(selected_list))
        # ----------------------Step 1

        filter_subsets = []
        for i in range(1, len(selected_list) + 1):
            filter_subsets += list(combinations(selected_list, i))
        filter_subsets = [list(tuples) for tuples in filter_subsets]
        # ----------------------Step 2

        counter1 = Counter(query_list)
        final_entities = [[]]
        for subset in filter_subsets:
            temp_list = " ".join(subset).split(" ")
            counter2 = Counter(temp_list)
            flag = 1
            for key, value in counter2.items():
                if key not in counter1.keys() or value > counter1[key]:
                    flag = 0
            if flag:
                final_entities.append(subset)
        # ----------------------Step 3

        query = defaultdict(int)
        count = 0
        first_element = defaultdict(int)
        first_element['tokens'] = query_list
        first_element['entities'] = []
        query[0] = dict(first_element)
        for ents in final_entities[1:]:
            count += 1
            entities_dic = defaultdict(int)
            query_list_dc = copy.deepcopy(query_list)
            words_list = " ".join(ents).split(" ")
            for word in words_list:
                query_list_dc.remove(word)
            entities_dic['tokens'] = query_list_dc
            entities_dic['entities'] = ents
            entities_dic = dict(entities_dic)
            query[count] = entities_dic
        query = dict(query)
        # ----------------------Step 4
        return query

    def max_score_query(self, query_splits, doc_id):
        tf_tokens_norm = defaultdict(int)
        tf_idf_token = defaultdict(int)
        for tokens, data in self.tf_tokens.items():
            if doc_id in data.keys():
                tf_tokens_norm[tokens] = 1.0 + math.log(1.0 + math.log(data[doc_id]))
                self.idf_tokens[tokens] = 1.0 + math.log(self.docs_num / (1.0 + len(data)))
                tf_idf_token[tokens] = tf_tokens_norm[tokens] * self.idf_tokens[tokens]
            else:
                tf_idf_token[tokens] = 0.0
        tf_idf_token = dict(tf_idf_token)
        tf_entities_norm = defaultdict(int)
        tf_idf_entity = defaultdict(int)
        for entities, data in self.tf_entities.items():
            if doc_id in data.keys():
                tf_entities_norm[entities] = 1.0 + math.log(data[doc_id])
                self.idf_entities[entities] = 1.0 + math.log(self.docs_num / (1.0 + len(data)))
                tf_idf_entity[entities] = tf_entities_norm[entities] * self.idf_entities[entities]
            else:
                tf_idf_entity[entities] = 0.0
        tf_idf_entity = dict(tf_idf_entity)

        maxscore_result = []
        max_score = 0
        for split_num, query in query_splits.items():
            s_tokens = 0
            s_entities = 0
            score = 0
            for tokens in query['tokens']:
                if tokens in tf_idf_token.keys():
                    s_tokens += tf_idf_token[tokens]
                else:
                    s_tokens += 0
            for entities in query['entities']:
                if entities in tf_idf_entity.keys():
                    s_entities += tf_idf_entity[entities]
                else:
                    s_entities += 0
            score = s_entities + 0.4 * s_tokens
            if score > max_score:
                max_score = score
                maxscore_result = []
                maxscore_result.append(max_score)
                maxscore_result.append(dict(query))
        maxscore_result = tuple(maxscore_result)
        return maxscore_result

