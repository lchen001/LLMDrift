import requests
import os
import time
import pickle
#import cohere
import json
from utils import compute_cost
import os
#import anthropic

import tiktoken

import wikipedia
from langchain import Wikipedia
from langchain.llms import AzureOpenAI, OpenAI
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.agents.react.base import DocstoreExplorer
from langchain.callbacks import get_openai_callback
from langchain.llms import Anthropic
import openai

#from mcli.sdk import predict


docstore=DocstoreExplorer(Wikipedia())
tools = [
    Tool(
        name="Search",
        func=docstore.search,
        description="useful for when you need to ask with search"
    ),
    Tool(
        name="Lookup",
        func=docstore.lookup,
        description="useful for when you need to ask with lookup"
    )

]

config_path ="../../config/serviceinfo.json"

#from transformers import CodeGenTokenizerFast
#tokenizer_FFAI = CodeGenTokenizerFast.from_pretrained("Salesforce/codegen-350M-mono")

from transformers import GPT2Tokenizer

tokenizer_FFAI = GPT2Tokenizer.from_pretrained("gpt2")


class GenerationParameter(object):
    def __init__(self,
                 max_tokens=100,
                 temperature=0.1,
                 stop=["\n"],
                 date="20230301",
                 trial = 0,
                 system_prompt=None,
                 ):
        self.max_tokens=max_tokens
        self.temperature=temperature
        self.stop = stop
        self.date = date
        self.system_prompt = system_prompt
        self.readable = dict()
        self.readable['max_tokens'] = max_tokens        
        self.readable['temperature'] = temperature
        self.readable['stop'] = stop
        self.readable['date'] = date
        self.readable['trial'] = trial
        if(self.system_prompt!=None):
            self.readable['system prompt'] = self.system_prompt

    def get_dict(self,
                 ):
        return self.readable 

class ModelService:
    """Base class for a model provider, currently we support API models, but this class can be subclassed with support
    for local GPU models"""
    def getcompletion(self,
                      context,
                      use_save=False,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(),
                      ):
        """
            Given a context, generate a text output
        """        
        raise NotImplementedError

class APIModelProvider(ModelService):
    """Provider that calculates conditional logprobs through a REST API"""
    _CONFIG = json.load(open("../../config/serviceinfo.json"))
    #print("config",_CONFIG)
    def getcompletion(self,
                      context,
                      use_save=False,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(),
                      ):
        endpoint = self._get_endpoint()
        req = self._request_format(context,
                                   genparams)
        self.context = context
        if(use_save==True):
            #print("-try previous call-",savepath)
            try:
                response = self.read_response(savepath)
                print('Have predicted before.')
                #print('prediction',self.mlservice.read_response(finalpath))
            except:
                print("new call")
                response = self._api_call( endpoint, data=req, api_key=self._API_KEY, retries=10, retry_grace_time=10)
                self.write_response(savepath)
        else:
            time1 = time.time()
            response = self._api_call( endpoint, data=req, api_key=self._API_KEY, retries=10, retry_grace_time=10)
            latency = time.time()-time1

        result = self._response_format(response)
        cost = self._get_cost(context,result)        
        result['cost'] = cost
        result['latency'] = latency

        return result

    def read_response(self, path="test"):
        f = open(path, 'rb')
        self.response = pickle.load(f)
        #self._response2pred(self.response)
        f.close()
        try:
            return self.response.json()
        except:
            return self.response
            
    def write_response(self,path="test"):
        filehandler = open(path, 'wb') 
        pickle.dump(self.response, filehandler)
        filehandler.close()
        try:
            return self.response.json()
        except:
            return self.response
        
    def _request_format(self,
                     genparams):
        raise NotImplementedError

    def _response_format(self,
                         response,
                         ):
        raise NotImplementedError
         
    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        for i in range(retries):
            #print("data is",data)
            #'''
            '''
            res = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=data,
                timeout=60)
            print("res is",res)
            print("made it")
            '''
            while True:
                try:
                    res = requests.post(
                endpoint,
                headers={"Authorization": f"Bearer {api_key}"},
                json=data,
                timeout=120)
                    res.raise_for_status()  # Raise an exception if the response is an HTTP error
                    break  # If we got a response, exit the loop
                except (requests.exceptions.Timeout, requests.exceptions.RequestException):
                    # If we got a timeout or another kind of error, retry the request
                    print("timeout, retry")
                    print("data is",data)
                    time.sleep(5)
                    continue
            #'''
            print("res is", res)
            if res.status_code == 200:
                print("succeed!")
                self.response = res
                return res.json()
            else:
                print("failed res is:",res)
            print(f"API call failed with {res}. Waiting {retry_grace_time} seconds")
            time.sleep(retry_grace_time)
        raise TimeoutError(f"API request failed {retries} times, giving up!") 

    def _get_cost(self,
                  context,
                  completion,
                ):
        tk1, tk2 = self._get_io_tokens(context,completion)
        #print("self._CONFIG[self._NAME]",self._CONFIG[self._NAME])
        cost = compute_cost(tk1,tk2,self._CONFIG[self._NAME][self._model])
        return cost
    
    def _get_endpoint(self):
        endpoint = self._ENDPOINT.format(engine=self._model)
        return endpoint
        
    def _get_token_num(self, text):
        
        enc = tiktoken.encoding_for_model(self._model)
        token_num = len(enc.encode(text))
        return token_num
                
class OpenAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("OPENAI_ENDPOINT", "https://api.openai.com/v1/engines/{engine}/completions")
    _API_KEY = os.environ.get('OPENAI_API_KEY', None)
    _NAME = "openai"
    
    def __init__(self, model):
        self._model = model
        self.temp_cmp = {'raw':{'usage':{'prompt_tokens':0}}}
        assert self._API_KEY is not None, "Please set OPENAI_API_KEY env var for running through OpenAI"

    def _request_format(self,
                        context,
                        genparams):
        #tk1,tk2 = self._get_io_tokens(context,self.temp_cmp)
        if(self._model in ['text-curie-001','text-babbage-001','text-ada-001']):
            #tk1 = len(context) # TODO: change this to real token count
            tk1 = len(tokenizer_FFAI(context)['input_ids'])
            print("length:",tk1)
            if(tk1+genparams.max_tokens>=2047):
                print("warning: the length is too large. only take the last few ones")
                context = context[-2048+genparams.max_tokens:]
        else:
            token = self._get_token_num(context)
            total_max = 3800
            if(token+genparams.max_tokens>=total_max):
                print("token number",token)
                max_token = total_max - token
            else:
                max_token = genparams.max_tokens
        print("max token",max_token)
        req = {
            "prompt": context,
            "echo": False,
            "max_tokens": max_token,
            "logprobs": 1,
            "temperature": genparams.temperature,
            "top_p": 1,
            "stop":genparams.stop,
        }
        return req
    
    def _response_format(self,
                         response,
                         ):
        result = dict()
        result['raw'] = response
        result["completion"] = response['choices'][0]['text']
        return result    
    
    def _get_io_tokens(self,context, completion):
        #print("completion raw:",completion['raw'])
        tk1 = completion['raw']['usage']['prompt_tokens']
        try:
            tk2 = completion['raw']['usage']['completion_tokens']
        except:
            tk2 = 0
            
        return tk1, tk2  
    
class OpenAIChatModelProvider_old(APIModelProvider):
    _ENDPOINT = os.environ.get("OPENAICHAT_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    _API_KEY = os.environ.get('OPENAI_API_KEY', None)
    _NAME = "openaichat"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set OPENAI_API_KEY env var for running through OpenAI"

    def _get_token_num(self, text):
        
        enc = tiktoken.encoding_for_model(self._model)
        token_num = len(enc.encode(text))
        return token_num

    def _request_format(self,
                        context,
                        genparams):
        # calculate the token size
        
        token = self._get_token_num(context)
        total_max = 3800
        if(token+genparams.max_tokens>=total_max):
            print("token number",token)
            max_token = total_max - token
        else:
            max_token = genparams.max_tokens
        print("max token",max_token)
        req = {
            "messages": [{"content":context,"role":"user"}],
            #"echo": False,
            "max_tokens": max_token,
            #"logprobs": 1,
            "temperature": genparams.temperature,
            #"top_p": 1,
            #"stop":, # seems it does not allow \n to be the stopping token.
            "model":self._model,
            #"role":"user",
        }
        #print("the request is---")
        #print(req)
        return req
    
    def _response_format(self,
                         response,
                         ):
        #print("response is",response)
        result = dict()
        result['raw'] = response
        result["completion"] = response['choices'][0]['message']['content']
        return result    
    
    def _get_io_tokens(self,context, completion):
        #print("completion raw:",completion['raw'])
        tk1 = completion['raw']['usage']['prompt_tokens']
        try:
            tk2 = completion['raw']['usage']['completion_tokens']
        except:
            tk2 = 0
            
        return tk1, tk2  

class AI21ModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("AI21_STUDIO_ENDPOINT", "https://api.ai21.com/studio/v1/{engine}/complete")
    _API_KEY = os.environ.get('AI21_STUDIO_API_KEY', None)
    _NAME = "ai21"
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set AI21_STUDIO_API_KEY env var for running through AI21 Studio"

    def _request_format(self,
                        context,
                        genparams):
        req = {
            "prompt": context,
            "maxTokens": genparams.max_tokens,
            "temperature": genparams.temperature,
            "stopSequences":genparams.stop,
        }           
        return req
    
    def _response_format(self,
                         response,
                         ):
        result = dict()
        result['raw'] = response
        result["completion"] = response['completions'][0]['data']['text']
        return result    
    
    def _get_io_tokens(self,context, completion):
        tk1 = len(completion['raw']['prompt']['tokens'])
        tk2 = len(completion['raw']['completions'][0]['data']['tokens'])
        return tk1, tk2 

class CohereAIModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("COHERE_STUDIO_ENDPOINT", "https://api.ai21.com/studio/v1/{engine}/complete")
    _API_KEY = os.environ.get('COHERE_STUDIO_API_KEY', None)
    _NAME = "cohere"

    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set COHERE_STUDIO_API_KEY env var for running through AI21 Studio"
    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        
        co = cohere.Client(api_key)
        try:
            response = co.generate( 
            model=self._model, 
            prompt=data['prompt'], 
            max_tokens=data['max_tokens'], 
            temperature=data['temperature'], 
            k=data['k'], 
            p=data['p'], 
            frequency_penalty=data['frequency_penalty'], 
            presence_penalty=data['presence_penalty'], 
            stop_sequences=data['stop_sequences'], 
            return_likelihoods=data['return_likelihoods']) 
        except:
            response = co.generate( 
            model=self._model, 
            prompt="test", 
            max_tokens=data['max_tokens'], 
            temperature=data['temperature'], 
            k=data['k'], 
            p=data['p'], 
            frequency_penalty=data['frequency_penalty'], 
            presence_penalty=data['presence_penalty'], 
            stop_sequences=data['stop_sequences'], 
            return_likelihoods=data['return_likelihoods']) 
            
        self.response = response
        return response

    def _request_format(self,
                        context,
                        genparams):
        req = {
            "prompt": context,
            "model":self._model,
            #"topKReturn": 10,
             "max_tokens":genparams.max_tokens, 
             "temperature":genparams.temperature, 
            "k":1,
            "num_generations":1,
            "p":1, 
            "frequency_penalty":0, 
            "presence_penalty":0, 
            "stop_sequences":genparams.stop, 
            "return_likelihoods":'ALL',          
        }
        return req
    
    def _response_format(self,
                         response,
                         ):
        fullresponse = dict()
        fullresponse['raw'] = ''

        try:
            #print("response is:", response)
            print("type of the response",type(response))
            token_likelihoods = response.generations[0].token_likelihoods
            text = [i1.token for i1 in token_likelihoods]
            text = "".join(text)
            #print("text is",text[len(context):])
            fullresponse["completion"] = response.generations[0].text
        except:
            fullresponse["completion"]=''
        print("completion:",fullresponse["completion"])
        return fullresponse
    
    def _get_io_tokens(self,context, completion):
        tk1 = len(context)/1000
        tk2 = len(completion)/1000
        return tk1, tk2
	
class ForeFrontAIModelProvider(APIModelProvider):
    _API_KEY = os.environ.get('FOREFRONT_API_KEY', None)  
    _NAME = "ffai"
    _ENDPOINT_MAP = {"QA":"https://shared-api.forefront.link/organization/nKKlZP3F37RN/codegen-16b-nl/completions/eGQdyiZlHIW4",
                     "CodeGen":"https://shared-api.forefront.link/organization/nKKlZP3F37RN/codegen-16b-nl/completions/eGQdyiZlHIW4",
                     "Pythia":"https://shared-api.forefront.link/organization/nKKlZP3F37RN/pythia-20b/completions/vanilla",
                     }
    def __init__(self, model="QA"):
        self._model = model
        assert self._API_KEY is not None, "Please set FOREFRONT_API_KEY env var for running through Forefront"  
    
    def _request_format(self,
                        context,
                        genparams):
#        if(self._model in ['QA','text-babbage-001','text-ada-001']):
        if(1):
            #tk1 = len(context) # TODO: change this to real token count
            tk1 = len(tokenizer_FFAI(context)['input_ids'])
            print("length:",tk1)
            if(tk1+genparams.max_tokens>=2047):
                print("warning: the length is too large. only take the last few ones")
                context = context[-2048+genparams.max_tokens:]
        
        req = {
            "text": context,
            "numResults": 1,
            "length": genparams.max_tokens,
            "topKReturn": 1,
            "temperature":genparams.temperature,
            "stop":["\n"],
            "logprobs":1,
            "echo":False,
        }
        return req
    
    def _response_format(self,
                         response,
                         ):
        result = dict()
        result["raw"] = response
        result['completion'] = response['result'][0]['completion']
        return result    
    
    def _get_io_tokens(self,context, completion):
        tk1 = len(tokenizer_FFAI(context)['input_ids'])
        tk2 = len(completion['raw']["logprobs"]['tokens'])
        return tk1, tk2
    
    def _get_endpoint(self):
        endpoint = self._ENDPOINT_MAP[self._model]
        return endpoint
			  
class TextSynthModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("TEXTSYNTH_ENDPOINT", "https://api.textsynth.com/v1/engines/{engine}/completions")
    _API_KEY = os.environ.get('TEXTSYNTH_API_SECRET_KEY', None)
    _NAME = "textsynth"

    def __init__(self, model="QA"):
        self._model = model
        assert self._API_KEY is not None, "Please set FOREFRONT_API_KEY env var for running through Forefront"
		
    def _request_format(self,
                        context,
                        genparams):
        req = {
            "prompt": context,
            "max_tokens": genparams.max_tokens,
            "temperature":genparams.temperature,
            "stop":genparams.stop,
        }
        return req
    
    def _response_format(self,
                         response,
                         ):
        result = dict()
        result['raw'] = response
        result['completion'] = result['raw']['text']
        return result    
    
    def _get_io_tokens(self,context, completion):
        tk1 = completion['raw']['input_tokens']
        tk2 = completion['raw']['output_tokens']
        return tk1, tk2

class AnthropicModelProvider(APIModelProvider):
    _ENDPOINT = os.environ.get("ANTHROPIC_ENDPOINT", "https://api.anthropic.com/v1/complete")
    _API_KEY = os.environ.get('ANTHROPIC_API_KEY', None)
    _NAME = "anthropic"
    
    def __init__(self, model):
        self._model = model
        assert self._API_KEY is not None, "Please set ANTHROPIC_API_KEY env var for running through OpenAI"
        self.client = anthropic.Client(os.environ['ANTHROPIC_API_KEY'])


    def _request_format(self,
                        context,
                        genparams):
        req = {
            "prompt": context,
            #"echo": False,
            "max_tokens_to_sample": genparams.max_tokens,
            #"logprobs": 1,
            "temperature": genparams.temperature,
            #"top_p": 1,
            #"stop":, # seems it does not allow \n to be the stopping token.
            "model":self._model,
            #"role":"user",
        }
        #print("the request is---")
        print(req)
        return req
    
    def _response_format(self,
                         response,
                         ):
        #print("response is",response)
        result = dict()
        result['raw'] = response
        result["completion"] = response['completion']
        return result    
    
    def _get_io_tokens(self,context, completion):
        #print("completion raw:",completion['raw'])
        tk1 = anthropic.count_tokens(context)
        tk2 = anthropic.count_tokens(completion['completion'])
        return tk1, tk2

    def _api_call(self, endpoint, data, api_key, retries=10, retry_grace_time=10):
        response = self.client.completion(
            prompt=anthropic.HUMAN_PROMPT+data['prompt']+anthropic.AI_PROMPT,
            model=data['model'],
            max_tokens_to_sample=data['max_tokens_to_sample'],
            )  
        return response      


class AgentOpenAIModelProvider(ModelService):
    _CONFIG = json.load(open("../../config/serviceinfo.json"))
    #print("config",_CONFIG)
    def __init__(self, model):
        self._model = model

    def getcompletion(self,
                      context,
                      use_save=False,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(),
                      ):
        # create model
        temperature = genparams.get_dict()['temperature']
        llm = OpenAI(model_name=self._model,temperature=temperature)
        # create agent
        react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=False)
        # run the agent
        time1 = time.time()
        try:
            with get_openai_callback() as cb:
                result_raw = react.run(context)
        except Exception as e:
            error = str(e)
            result_raw = error
        latency = time.time()-time1
        # format
        result = dict()
        result['cost'] = cb.total_cost
        result['latency'] = latency
        result['raw'] = result_raw
        result['completion'] = result_raw
        return result


class AgentAnthropicModelProvider(ModelService):
    _CONFIG = json.load(open("../../config/serviceinfo.json"))
    #print("config",_CONFIG)
    def __init__(self, model):
        self._model = model

    def getcompletion(self,
                      context,
                      use_save=False,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(),
                      ):
        # create model
        temperature = genparams.get_dict()['temperature']
        llm = Anthropic(model=self._model,temperature=temperature)
        # create agent
        react = initialize_agent(tools, llm, agent=AgentType.REACT_DOCSTORE, verbose=False)
        # run the agent
        time1 = time.time()
        try:
            with get_openai_callback() as cb:
                result_raw = react.run(context)
        except Exception as e:
            error = str(e)
            result_raw = error
        latency = time.time()-time1
        # format
        result = dict()
        result['cost'] = cb.total_cost
        result['latency'] = latency
        result['raw'] = result_raw
        result['completion'] = result_raw
        return result

        endpoint = self._get_endpoint()
        req = self._request_format(context,
                                   genparams)
        self.context = context
        if(use_save==True):
            #print("-try previous call-",savepath)
            try:
                response = self.read_response(savepath)
                print('Have predicted before.')
                #print('prediction',self.mlservice.read_response(finalpath))
            except:
                print("new call")
                response = self._api_call( endpoint, data=req, api_key=self._API_KEY, retries=10, retry_grace_time=10)
                self.write_response(savepath)
        else:
            time1 = time.time()
            response = self._api_call( endpoint, data=req, api_key=self._API_KEY, retries=10, retry_grace_time=10)
            latency = time.time()-time1

        result = self._response_format(response)
        cost = self._get_cost(context,result)        
        result['cost'] = cost
        result['latency'] = latency

        return result

class AzureOpenAIModelProvider(APIModelProvider):
    """Provider that calculates conditional logprobs through a REST API"""
    _CONFIG = json.load(open(config_path))
    _NAME = "azureopenai"
    #print("config",_CONFIG)
    def __init__(self, model):
        self._model = model
        self._setup()
        return 

    def _setup(self,):
        openai.api_type = "azure"
        #openai.api_base = "https://llmaigpt4.openai.azure.com/"
        openai.api_base = "https://gpt4v222.openai.azure.com/"
        openai.api_version = "2023-03-15-preview"
        openai.api_key = os.getenv("AZURE_OPENAI_API_KEY")
        self.openai = openai

    def getcompletion(self,
                      context,
                      use_save=False,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(),
                      ):

        #Note: The openai-python library support for Azure OpenAI is in preview.
        # form message
        time1 = time.time()
        message = []
        if('system prompt' in genparams.get_dict()):
            sys_prompt = genparams.get_dict()['system prompt']
            if(sys_prompt!=""):
                message.append({"role":"system","content":sys_prompt})
        message.append({"role":"user","content":context})

        # form params
        temperature = genparams.get_dict()['temperature']
        max_tokens = genparams.max_tokens
        stop = genparams.stop
        #print("message is::",message)
        '''
        response = self.openai.ChatCompletion.create(
            engine=self._model,
            messages = message, #[{"role":"user","content":"Is 6547 a prime number? Think step by step and then answer \"[Yes]\" or \"[No]\"."}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop)
        print("azure response is",response)
        '''
        while(1):
            try:
                response = self.openai.ChatCompletion.create(
            engine=self._model,
            messages = message, #[{"role":"user","content":"Is 6547 a prime number? Think step by step and then answer \"[Yes]\" or \"[No]\"."}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop)
                break
            except:
                print("the api failed. Wait and Retry in 10s")
                time.sleep(10)
        
        latency = time.time()-time1
        result = {}        
        cost = self._get_cost(context,response)        
        result['cost'] = cost        
        result['latency'] = latency
        result['raw'] = response.to_dict()
        result['completion'] = response['choices'][0]['message']['content']

        return result 
    def _get_io_tokens(self,context,completion):
        tk1 = completion['usage']['prompt_tokens']
        tk2 = completion['usage']['completion_tokens']
        return tk1, tk2

class OpenAIChatModelProvider(APIModelProvider):
    """Provider that calculates conditional logprobs through a REST API"""
    _CONFIG = json.load(open(config_path))
    _ENDPOINT = os.environ.get("OPENAICHAT_ENDPOINT", "https://api.openai.com/v1/chat/completions")
    _API_KEY = os.environ.get('OPENAI_API_KEY', None)
    _NAME = "openaichat"

    #print("config",_CONFIG)
    def __init__(self, model):
        self._model = model
        self._setup()
        return 

    def _setup(self,):
        #openai.api_type = "open_ai"
        #openai.api_base = "https://llmaigpt4.openai.azure.com/"
        #openai.api_base = 'https://api.openai.com/v1'
        #openai.api_version = "2023-03-15-preview"
        openai.api_base = "https://api.openai.com/v1"
        openai.api_version = None
        openai.api_type = 'open_ai'        
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.openai = openai

    def getcompletion(self,
                      context,
                      use_save=False,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(),
                      ):

        #Note: The openai-python library support for Azure OpenAI is in preview.
        # form message
        time1 = time.time()
        message = []
        if('system prompt' in genparams.get_dict()):
            sys_prompt = genparams.get_dict()['system prompt']
            if(sys_prompt!=""):
                message.append({"role":"system","content":sys_prompt})
        message.append({"role":"user","content":context})

        # form params
        temperature = genparams.get_dict()['temperature']
        max_tokens = genparams.max_tokens
        stop = genparams.stop
        #print("message is::",message)
        '''
        response = self.openai.ChatCompletion.create(
            engine=self._model,
            messages = message, #[{"role":"user","content":"Is 6547 a prime number? Think step by step and then answer \"[Yes]\" or \"[No]\"."}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop)
        print("azure response is",response)
        '''
        while(1):
            try:
                response = self.openai.ChatCompletion.create(
            model=self._model,
            messages = message, #[{"role":"user","content":"Is 6547 a prime number? Think step by step and then answer \"[Yes]\" or \"[No]\"."}],
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop)
                break
            except:
                print("the api failed. Wait and Retry in 10s")
                time.sleep(10)
        
        latency = time.time()-time1
        result = {}        
        cost = self._get_cost(context,response)        
        result['cost'] = cost        
        result['latency'] = latency
        result['raw'] = response.to_dict()
        result['completion'] = response['choices'][0]['message']['content']

        return result 
    def _get_io_tokens(self,context,completion):
        tk1 = completion['usage']['prompt_tokens']
        tk2 = completion['usage']['completion_tokens']
        return tk1, tk2


class LLama2ChatModelProvider(APIModelProvider):
    """Provider that calculates conditional logprobs through a REST API"""
    _CONFIG = json.load(open(config_path))
    _ENDPOINT = os.environ.get("MOSAICLLAMA2_ENDPOINT", "https://models.hosted-on.mosaicml.hosting/llama2-70b-chat/v1")
    #_API_KEY = os.environ.get('OPENAI_API_KEY', None)
    _NAME = "llama2chat"

    #print("config",_CONFIG)
    def __init__(self, model):
        self._model = model
        #self._setup()
        return 

    def _message2prompt(self,message):
        prompt = ""
        for item in message:
            temp = ''
            if(item['role']=='system'):
                temp = '<<SYS>> {content} <</SYS>> '.format(content=item['content'])
            if(item['role']=='user'):
                temp = item['content']
            prompt+=temp
        prompt = '<s>[INST] {prompt} [/INST]'.format(prompt=prompt)
        return prompt

    def getcompletion(self,
                      context,
                      use_save=False,
                      savepath="raw.pkl",
                      genparams = GenerationParameter(),
                      ):

        #Note: The openai-python library support for Azure OpenAI is in preview.
        # form message
        time1 = time.time()
        message = []
        if('system prompt' in genparams.get_dict()):
            sys_prompt = genparams.get_dict()['system prompt']
            if(sys_prompt!=""):
                message.append({"role":"system","content":sys_prompt})
        message.append({"role":"user","content":context})

        # form params
        temperature = genparams.get_dict()['temperature']
        max_tokens = genparams.max_tokens
        stop = genparams.stop
        #print("message is::",message)
        prompt = self._message2prompt(message)
        #print("prompt is---",prompt)
        '''
        response = predict(self._ENDPOINT, {"inputs": [prompt], "parameters": {'temperature': temperature,'max_new_tokens':max_tokens}})
        '''
        while(1):
            try:
                response = predict(self._ENDPOINT, {"inputs": [prompt], "parameters": {'temperature': temperature,'max_new_tokens':max_tokens}})
                #print("response is---",response)

                break
            except:
                print("the api failed. Wait and Retry in 10s")
                time.sleep(10)
        
        latency = time.time()-time1
        result = {}        
        cost = self._get_cost(context,response)        
        result['cost'] = cost        
        result['latency'] = latency
        result['raw'] = response
        result['completion'] = response['outputs'][0]

        return result 
    def _get_io_tokens(self,context,completion):
        tk1 = len(context)
        tk2 = len(completion['outputs'][0])
        return tk1, tk2


_PROVIDER_MAP = {
    "openai": OpenAIModelProvider, # cleaned
    "ai21": AI21ModelProvider, # cleaned
    "cohere":CohereAIModelProvider, # cleaned
    "forefrontai":ForeFrontAIModelProvider, # cleaned
    "textsynth":TextSynthModelProvider, # cleaned
    "openaichat":OpenAIChatModelProvider,# cleaned
    "llama2chat":LLama2ChatModelProvider,# cleaned
    "anthropic":AnthropicModelProvider,
    "agent_openai":AgentOpenAIModelProvider,
    "agent_anthropic":AgentAnthropicModelProvider,
    "azure_openai":AzureOpenAIModelProvider,
}


def make_model(provider, model):
    assert provider in _PROVIDER_MAP, f"No model provider '{provider}' implemented"
    return _PROVIDER_MAP[provider](model)
