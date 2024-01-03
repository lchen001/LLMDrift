from modelservice import make_model,GenerationParameter
from sqlitedict import SqliteDict
import json
from transformers import GPT2Tokenizer
#import settings
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

global mydict

def form_keys(service_id,
              genparams,
              query,
              ):
    mydict = genparams.get_dict()
    mydict['service_id'] = service_id
    mydict['query'] = query
    return repr(mydict)
    
class LLMEngine(object):
    def __init__(self, 
                 service_name=list(),
                 db_path="",
                 ):
        global mydict
        mydict = SqliteDict(db_path, autocommit=True)
        self.service_name = service_name
        self.services = dict()
        self.cost = 0
        for item in service_name:
            provider = item.split('/')[0]
            name = item.split('/')[1]
            self.services[item] = make_model(provider,name)
        return
    
    def get_cost(self,):
        return self.cost
    
    def compute_cost(self,
                     input_text,
                     output_text,
                     service_name):
        cost = 0
        input_size = len(tokenizer(input_text)['input_ids'])
        gen_size = len(tokenizer(output_text)['input_ids']) 
        
        service_all = self.service_name
    
        
        cost_output = service_all[service_name]["cost_output"]
        cost_input = service_all[service_name]["cost_input"]
        cost_fixed = service_all[service_name]["cost_fixed"]
        fixed_size = service_all[service_name]["fixed_size"]
        
        cost = cost_input*input_size + cost_fixed
        if(gen_size>fixed_size):
            cost += cost_output*(gen_size - fixed_size)    
        
        return cost 

    def costestimate(self,
                     query:str,
                     service_name = "20000",
                     Params = GenerationParameter(max_tokens=50, 
                                                   temperature=0.1,
                                                   stop=["\n"],
                      )
                     ):
        max_token = Params.max_tokens
        cost = self.compute_cost(input_text = query,
                                 output_text = 'token '*max_token,
                                 service_name=service_name,) 
        
        return cost
    def getcompletion(self, 
                      query:str,
                      service_name = "20000",
                      use_save=False,
                      use_db = True,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(max_tokens=50, 
                                                   temperature=0.1,
                                                   stop=["\n"],
),  
                      ):
        model = self.services[service_name]
        key = service_name+"_"
        key = form_keys(service_id=service_name, genparams=genparams, query=query)
        if((key in mydict )and ('cost' in mydict[key]) ):
            #print("use cached!")
            completion = mydict[key]
        else:
            print("---invoke new API call---")
            print(service_name,mydict)
            #print(query)
            if(1):
                completion = model.getcompletion(query,
                                         use_save=use_save,
                                         genparams=genparams)  
            mydict[key] = completion
     
        if(use_db==False):
            completion = model.getcompletion(query,
                                         use_save=use_save,
                                         genparams=genparams)
            mydict[key] = completion
            print("do not use stored results!")
        try:
            cost =completion['cost']
        except:
            cost = self.compute_cost(input_text = query,
                                 output_text = completion['completion'],
                                 service_name=service_name,)    
        self.cost = cost
        self.completion = completion  
        self.latency = completion['latency']    
        return completion['completion']
    
    def get_last_cost(self,
                      ):
        #print("completion with cost",self.completion)
        return self.completion['cost']
    
    def get_latency(self):
        return self.latency

    def reset(self,
              ):
        self.cost = 0
        self.completion = None
