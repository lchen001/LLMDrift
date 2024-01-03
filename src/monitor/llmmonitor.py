import json, sys, pandas, os
sys.path.append('../service/')
from datetime import datetime
from llmengine import LLMEngine
from modelservice import GenerationParameter

def configparse(config):
    query_params = config['query_params']
    model_list = config['model_list']
    model_params = config['model_params']
    trail = config['trail']
    if('system_prompts' in config):
        system_prompts = config['system_prompts']
    else:
        system_prompts = [None]
    print("config parse system_prompts",system_prompts,config)
    return query_params, model_list,model_params,trail,system_prompts

class LLMMonitor(object):
    def __init__(self,):
        return
    
    def evaluate(self,
                 query_params={},
                 model_list = [],
                 model_params = [],
                 trials=1,
                 system_prompts=[None],
                 forcetime="",
                ):
        self.forcetime=forcetime
        # create the LLM engine
        MyLLMEngine = LLMEngine(service_name=model_list,
                 db_path="../../db/db_azure_test.sqlite")        
        # create queries
        queries = json.load(open(query_params['querypath']))
        # create answers
        answers = json.load(open(query_params['answerpath']))

        results = list()
        # iterate over model, query, trails, and system prompt
        # for each query and each model, generate the answer 
        for model in model_list:
            for idx, query in enumerate(queries):
                for i in range(trials):
                    for system_prompt in system_prompts:
                        print("system_prompt",system_prompt)
                        # create the GenerationParameter
                        genparams = self._create_genparams(
                             model_params=model_params,
                              trial=i,
                              system_prompt=system_prompt,
                         )
                        # create head fields
                        head = self._create_head(query_params=query_params,
                                            model=model,
                                             model_params = model_params,
                                            _id=query['_id'],
                                            query=query['query'],
                                            ref_answer=answers[idx]['answer'],
                                            trail=i,
                                            system_prompt=system_prompt,
                                            )
                        # create generation feilds
                        result = self.evaluate_once(MyLLMEngine,model,query['query'],answers[idx]['answer'],genparams)
                        head.update(result)
                        results.append(head)
        # save to the original path
        self.savepath(eval_result = pandas.DataFrame(results),path = query_params['savepath'])
        return pandas.DataFrame(results)
    
    def evaluate_once(self,MyLLMEngine,model,query,answer,genparams):
        #print("query",query,"genparams",genparams.get_dict())
        res = MyLLMEngine.getcompletion(query=query,service_name=model,genparams=genparams)
        latency = MyLLMEngine.get_latency()
        cost = MyLLMEngine.get_cost()
        now = datetime.now() # current date and time
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d")
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        if(self.forcetime==""):
            date1 = year+month+day
        else:
            date1 = str(self.forcetime)
        print("date1",date1)
        result = {'answer':res,
                 'latency':latency,
                 'cost':cost,
                 'date':date1,
                 'timestamp':date_time}

        return result

    def savepath(self,eval_result,path):
        # if file does not exist write header 
        if not os.path.isfile(path):
            eval_result.to_csv(path,escapechar='\\')
        else: # else it exists so append without mentioning the header
           eval_result.to_csv(path, mode='a', header=False, escapechar='\\')
        #eval_result.to_csv(query_params['savepath'],mode=mode)
        return
        
    def _create_head(self,
                     query_params={},
                     model="",
                 model_params = [],
                    _id="0",
                    query="",
                    ref_answer="",
                    trail=0,
                    system_prompt=None,):
        head = {'model':model,
                  'temperature':model_params['temperature'],
                  'max_tokens':model_params['max_tokens'],
                  'trail':trail,
                  'dataset':query_params['dataset_name'], 
                  'id':_id,
                  'query':query, 
                  'ref_answer':ref_answer,
                  'system_prompt':system_prompt,
                 }        
        return head
        
    def _create_columns(self,):
        df = pandas.DataFrame(columns=['model','temperature','max_tokens','trial','dataset', 'id', 'query', 'ref_answer','answer','latency','date','timestamp'])
        return df
    
    def _create_genparams(self,
                         model_params,
                          trial,
                          system_prompt,
                         ):
        now = datetime.now() # current date and time
        year = now.strftime("%Y")
        month = now.strftime("%m")
        day = now.strftime("%d") 
        date = year+month+day
        #print("type of date",type(date))
        if(self.forcetime==""):
            date = year+month+day
        else:
            #print("force time")
            date = self.forcetime     
        genparams = GenerationParameter(max_tokens=model_params['max_tokens'], 
                                temperature=model_params['temperature'],
                                stop=model_params['stop'],
                                date=date,
                                trial=trial,
                                system_prompt=system_prompt,
        )
        return genparams
        