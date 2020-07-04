## Import Libraries and Modules here...

import spacy
from collections import defaultdict
from collections import Counter
from itertools import combinations
import math
nlp = spacy.load("en_core_web_sm")



class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = {}
        self.tf_entities = {}

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = {}
        self.idf_entities = {}
        self.nb_of_docs = 0
        
    ## Your implementation for indexing the documents...
    def index_documents(self, documents):
        
        ## Replace this line with your implementation...
        self.nb_of_docs = len(documents)
        self.tf_entities = {}
        doc_id_dict = {}

        #entity
        for i in documents.keys(): 
        # traverse the documents, the key is the document id,the value is the content 
            entity_list = []
            #set an entity_list to store all the entity name of current document
            doc = nlp(documents[i])
            for word in doc.ents:
            #find the word which can be used as the key of final entity dictionary
                entity_list.append(word.text)
            c = Counter(entity_list)
            # count the frequency of all words 
            c_dict = dict(c)
            # transfer the Counter into a dictionary e.g.{'Trump':1}
            copy_c_dict = c_dict.copy()
            # copy the c_dict to avoid the problems of changing the c_dict
            doc_id_dict = {}
            #initialize the doc_id_dict 
            for key in copy_c_dict:
                self.tf_entities2 = self.tf_entities.copy()
                #do not change the self.tf_entities2 when the self.tf_entities is changed
                if key not in self.tf_entities2:
                #if the key not exist in the self.tf_entities2, add the key-value into the final entity dictionary 
                    doc_id_dict[i] = c_dict[key]
                    # assign the value e.g. {1:1} 
                    doc_id_dict2 = doc_id_dict.copy()
                    c_dict[key] = doc_id_dict2
                    #change the c_dict into {'Trump':{1:1}}
                    self.tf_entities[key] = c_dict[key]
                    #add new key-value into the final entity dictionary {'Trump':{1:1}}
                elif key in self.tf_entities2:
                #if entity has already existed, update the the entity valuee.g.{1:1}->{1:1,2:1}
                    temp_dict = {}
                    temp_dict = self.tf_entities[key].copy()
                    # change the inner dictionary{1:1} -> {1:1,2:1}
                    temp_dict[i] = c_dict[key]
                    # create a new key-value
                    self.tf_entities[key] = temp_dict  

        #token
        self.tf_tokens = {}
        for i in documents.keys():
            doc = nlp(documents[i])
            token_list =[]
            for token in doc:
                if token.is_stop != True and token.is_punct != True:
                    if token.ent_iob == 3 and token.text in self.tf_entities:
                        continue
                    token_list.append(token.text)

            token_list2 = token_list.copy()
            count = Counter(token_list)
            count_dict = dict(count)
            count_dict2 = count_dict.copy()
            d_id_dict = {}

            for key in count_dict2:
                self.tf_tokens2 = self.tf_tokens.copy()
                if key not in self.tf_tokens2:
                    d_id_dict[i] = count_dict[key]
                    d_id_dict2= d_id_dict.copy()
                    count_dict[key] = d_id_dict2
                    self.tf_tokens[key] = count_dict[key]
                elif key in self.tf_tokens2:
                    tmp_dict = {}
                    tmp_dict = self.tf_tokens[key].copy()
                    tmp_dict[i] = count_dict[key]
                    self.tf_tokens[key] = tmp_dict


    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):
        all_single_e = []
        #the different entity
        all_t_e = []
        #all the combination of entity, tuple 
        all_e = []
        #all the combination of entity, list 
        Q_dict = {}
        Q_list = Q.split()
        # the list of single word of Q(Query)
        # e.g. Q_list = ['New', 'York', 'Times', 'Trump', 'travel']
        # print(Q_list)
        for key in DoE:
            key_list = key.split()
            # key_list =['New', 'York', 'Times']
            #if the word in DoE has existed in Q, it is an entity
            for word in key_list:
                if word not in Q_list:
                    break
            else:
                if key not in all_single_e:
                    all_single_e.append(key)
        #e.g. all_single_e = ['New York Times', 'New York']

        #using combinations to computer all the possibilities
        #e.g. all_e =[['New York Times'], ['New York'], ['New York Times', 'New York']]
        Length = len(all_single_e)
        count = 1
        while(count <= Length):    
            all_t_e.extend(list(combinations(all_single_e,count)))
            count += 1
        for x in all_t_e:
            all_e_x = list(x)
            all_e.append(all_e_x)

        #according to the all_e to get the all_k
        #drop all the entities from the Q_list
        #e.g. all_k = [['Trump', 'travel'], ['Times', 'Trump', 'travel'], ['Trump', 'travel']]
        all_k = []
        for i in range(len(all_e)):
            ith_k = Q_list.copy()
            for j in range(len(all_e[i])):
                temp_list = all_e[i][j].split()         
                for r in temp_list:
                    if r in ith_k:
                        ith_k.remove(r)
            all_k.append(ith_k) 

        #format the e1 to avoid the invalid entities
        for x in reversed(range(len(all_e))):
            sum_e_k = []    
            sum_e_x = []

            for y in range(len(all_e[x])):
                strxy = "".join(all_e[x][y])
                #print(strxy)
                listxy = strxy.split()
                #print(listxy)
                sum_e_x.append(listxy)

            #Add all the ei and ki, if the number exceeds Q, delete ei and ki from all_e and all_k
            for ith in range(len(sum_e_x)):
                for jth in range(len(sum_e_x[ith])):
                    sum_e_k.append(sum_e_x[ith][jth])
            for ki in range(len(all_k[x])):
                sum_e_k.append(all_k[x][ki])
            #delete the invalid entity
            if len(sum_e_k) > len(Q_list):
                del all_e[x]
                del all_k[x]
        #add another condition e=[], k=Q_list     
        all_e.insert(0,[])
        all_k.insert(0,Q_list)
        # all_e 2 = [[], ['New York Times'], ['New York']]
        # all_k 2 = [['New', 'York', 'Times', 'Trump', 'travel'], ['Trump', 'travel'], ['Times', 'Trump', 'travel']]
        query_splits = {}
        for n in range(len(all_e)):
            query_splits[n] = {'tokens': all_k[n], 'entities': all_e[n]}
        return query_splits


    ## Your implementation to return the max score among all the query splits...
    def max_score_query(self, query_splits, doc_id):
        TF_IDF_t = {}
        TF_IDF_e = {}
        
        for tokens, t_times in self.tf_tokens.items():
            if doc_id in t_times.keys():
                TF_token = t_times[doc_id]
                TF_token_normal = 1.0 + math.log(1.0 + math.log(TF_token))
                self.idf_tokens[tokens] = 1.0 + math.log(self.nb_of_docs / (1.0 + len(t_times)))
                TF_IDF_t[tokens] = TF_token_normal * self.idf_tokens[tokens]
                
            else:
                TF_IDF_t[tokens] = 0.0
    
        for entities, e_times in self.tf_entities.items():
            if doc_id in e_times.keys():
                TF_entity = e_times[doc_id]
                TF_entity_normal = 1.0 + math.log(TF_entity)
                self.idf_entities[entities] = 1.0 + math.log(self.nb_of_docs / (1.0 + len(e_times)))
                TF_IDF_e[entities] = TF_entity_normal * self.idf_entities[entities]
            else:
                TF_IDF_e[entities] = 0.0
        
        max_score_dict = defaultdict(float)
        max_score_dict = {}
        max_score = float(0.0)
        
        for i in query_splits.keys():
            sum_tokens = float(0.0)
            sum_entites = float(0.0)            
            for e_items in query_splits[i]['entities']:
                if e_items not in TF_IDF_e.keys():
                    sum_entites = sum_entites
                else:
                    sum_entites = sum_entites + TF_IDF_e[e_items]    
            for t_items in query_splits[i]['tokens']:   
                if t_items not in TF_IDF_t.keys():
                    sum_tokens = sum_tokens
                else:
                    sum_tokens = sum_tokens + float(TF_IDF_t[t_items])
            score =  sum_entites + 0.4 * sum_tokens
            if score > max_score:
                max_score = score
                query_splits_str = str(query_splits[i])  
                max_score_dict[query_splits_str] = max_score
        for key,value in max_score_dict.items():
            if value == max_score:
                result = (value, eval(key))
        return result  
    
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})
        
