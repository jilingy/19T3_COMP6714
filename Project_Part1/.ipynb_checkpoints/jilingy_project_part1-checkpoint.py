## Import Libraries and Modules here...
import spacy
import math
import copy
from itertools import combinations
from collections import defaultdict,Counter
nlp = spacy.load('en_core_web_sm')

class InvertedIndex:
    def __init__(self):
        ## You should use these variable to store the term frequencies for tokens and entities...
        self.tf_tokens = defaultdict(dict)
        self.tf_entities = defaultdict(dict)

        ## You should use these variable to store the inverse document frequencies for tokens and entities...
        self.idf_tokens = defaultdict(dict)
        self.idf_entities = defaultdict(dict)
        self.docs_num = int
    
    ## Your implementation for indexing the documents...
    def index_documents(self, documents):   
        self.docs_num=len(documents)
        for key, value in documents.items():
            doc = nlp(value) 
            for token in doc:
                if not token.is_stop and not token.is_punct:
                    if token.text in self.tf_tokens.keys():
                        if key not in self.tf_tokens[token.text].keys():
                            newdict = defaultdict(int)
                            newdict[key]=1
                            self.tf_tokens[token.text].update(newdict)
                        else:
                            self.tf_tokens[token.text][key]+=1
                    elif token.text not in self.tf_tokens.keys():
                        smalldict = defaultdict(int)
                        smalldict[key]=1       
                        smalldict=dict(smalldict)
                        self.tf_tokens[token.text]=smalldict   
        
            for ent in doc.ents:  
                if ent.text in self.tf_entities.keys():
                    if key not in self.tf_entities[ent.text].keys():
                        newdict = defaultdict(int)
                        newdict[key]=1
                        self.tf_entities[ent.text].update(newdict)
                    else:
                        self.tf_entities[ent.text][key]+=1
                elif ent.text not in self.tf_entities.keys():
                    smalldict = defaultdict(int)
                    smalldict[key]=1       
                    smalldict=dict(smalldict)
                    self.tf_entities[ent.text]=smalldict 
                if ent.text in self.tf_tokens.keys():
                    self.tf_tokens[ent.text][key]-=1
                    if self.tf_tokens[ent.text][key]==int(0):
                        del self.tf_tokens[ent.text][key]
                    if self.tf_tokens[ent.text]=={}:
                        del self.tf_tokens[ent.text]  
        #print("self.tf_tokens",self.tf_tokens)
        #print("self.tf_entities",self.tf_entities)
        
        
    ## Your implementation to split the query to tokens and entities...
    def split_query(self, Q, DoE):  
        query_list=Q.split(" ")
        selected_list=[]
        for i in range(1,len(query_list)+1):
            comb_list=list(combinations(query_list, i)) 
            for item in comb_list:               
                word = " ".join(item)
                if word in DoE.keys():
                    selected_list.append(word)
        selected_list=list(set(selected_list))
        print("selected_list",selected_list)
        #print("\n")
        #----------------------以上是𝐒𝐭𝐞𝐩 1

        Filter_subsets = []
        for i in range(1,len(selected_list)+1):
            Filter_subsets+=list(combinations(selected_list , i))
        Filter_subsets=[list(tuples) for tuples in Filter_subsets]
        print("Filter_subsets:",Filter_subsets)
        #print("\n")
        #----------------------以上是𝐒𝐭𝐞𝐩 2
        
        counter1=Counter(query_list)
        final_entities=[[]]
        for subset in Filter_subsets:
            temp_list=" ".join(subset).split(" ")
            counter2=Counter(temp_list)
            flag=1
            for key,value in counter2.items():
                #print("key,value",key,value)
                if key not in counter1.keys() or value > counter1[key]:
                    flag=0
            if flag:
                final_entities.append(subset)   
        
        #print("final_entities",final_entities)
        #----------------------以上是𝐒𝐭𝐞𝐩 3
        
        query = defaultdict(int)
        count=0
        first_element=defaultdict(int)
        first_element['tokens'] = query_list
        first_element['entities'] = []  
        query[0]=dict(first_element)
        for ents in final_entities[1:]:
            count+=1
            entitiess_dic = defaultdict(int)
            query_list_dc=copy.deepcopy(query_list)
            words_list = " ".join(ents).split(" ")
            #print("words_list",words_list)
            for word in words_list:
                query_list_dc.remove(word)
            entitiess_dic['tokens']=query_list_dc
            entitiess_dic['entities']=ents
            entitiess_dic=dict(entitiess_dic)
            query[count]=entitiess_dic  
        query=dict(query)
        print(query)
        #----------------------以上是𝐒𝐭𝐞𝐩 4
        return query
    
        
    ## Your implementation to return the max score among all the query splits...w
    def max_score_query(self, query_splits, doc_id):        
        ##----------------------TF-IDF index For Tokens:
        tf_tokens_norm=defaultdict(int)
        tf_idf_token=defaultdict(int)
        for tokens, data in self.tf_tokens.items():
            if doc_id in data.keys():
                tf_tokens_norm[tokens]=1.0+math.log(1.0+math.log(data[doc_id]))
                self.idf_tokens[tokens]=1.0+math.log(self.docs_num/(1.0+len(data)))
                tf_idf_token[tokens]=tf_tokens_norm[tokens] * self.idf_tokens[tokens]
            else:
                tf_idf_token[tokens]=0.0
        tf_idf_token=dict(tf_idf_token)
        #print("----------------------tf_idf_token: ",tf_idf_token)
        #print("\n")
        
        tf_entities_norm=defaultdict(int)
        tf_idf_entity=defaultdict(int)
        for entities, data in self.tf_entities.items():
            if doc_id in data.keys():
                tf_entities_norm[entities]=1.0 + math.log(data[doc_id])
                self.idf_entities[entities]=1.0 + math.log(self.docs_num/(1.0 + len(data)))
                tf_idf_entity[entities] = tf_entities_norm[entities] * self.idf_entities[entities]
            else:
                tf_idf_entity[entities]=0.0
              
        tf_idf_entity=dict(tf_idf_entity)
        #print("----------------------tf_idf_entity",tf_idf_entity)
        #print("\n")
        
        maxscore_result=[]
        max_score=0
        for split_num,query in query_splits.items():
            s_tokens = 0
            s_entities = 0
            score = 0
            for tokens in query['tokens']:
                if tokens in tf_idf_token.keys():
                    s_tokens+=tf_idf_token[tokens]
                else:
                    s_tokens+=0     
            for entities in query['entities']:
                if entities in tf_idf_entity.keys():
                    s_entities+=tf_idf_entity[entities]
                else:
                    s_entities+=0
            score = s_entities + 0.4 * s_tokens
            if score > max_score:   
                max_score=score
                maxscore_result=[]
                maxscore_result.append(max_score)
                maxscore_result.append(dict(query))
        maxscore_result=tuple(maxscore_result)    
        return maxscore_result
                    
        ## Replace this line with your implementation...
        ## Output should be a tuple (max_score, {'tokens': [...], 'entities': [...]})

