# -*- coding: utf-8 -*-
"""
Created on Mon July  3 21:06:55 2023
@author: Lingjiao Chen
"""
import re, string

def calculate_score(row):
    # Perform scoring logic based on 'answer' and 'ref_answer'
    # For example, you can use a simple equality check and assign a score of 1 for a match, and 0 for a mismatch
    score = evaluate(row['answer'], row['ref_answer'])
    #print("lens",len(row['answer']),len(row['ref_answer']))
    return score

def calculate_contain_score(row):
    # Perform scoring logic based on 'answer' and 'ref_answer'
    # For example, you can use a simple equality check and assign a score of 1 for a match, and 0 for a mismatch
    #score = evaluate(row['answer'], row['ref_answer'])
    #print("lens",len(row['answer']),len(row['ref_answer']))
    sub1 = row['ref_answer'].lower()
    full = row['answer'].lower()

    def add_square_brackets(string):
        words = string.split()
        words_with_brackets = [f"[{word}]" for word in words]
        result = " ".join(words_with_brackets)
        return result

    if('['+sub1+']' in add_square_brackets(full)):
        #print("contained!")
        score = 1
    else:
        score=0

    return score

def eval_code(datapoint):
    #time.sleep(1)
    # get filename
    title = datapoint['id'].split("-", 1)[1] 
    # get code
    code = datapoint['answer']
    # save to filename
    with open('temp/'+title+'.py', 'w') as f:  
        f.write(code)
    # submit code
    command = 'leetcode submit temp/'+title+'.py'
    while(1):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)
        stdout, stderr = process.communicate()
        # get results
        Out1 = stdout.decode()
        print("Output:", Out1)
        Error1 = stderr.decode()
        #print("Error:", Error1 )
        if('http error' in Out1):
            print("sleep and retry")
            time.sleep(5)
        else:
            break
    return Out1

def direct_usable(datapoint):
    if ('Accepted' in datapoint['Code_Submit']):
        return 1
    if ('Error' in datapoint['Code_Submit']):
        return 0
    if ('Wrong' in datapoint['Code_Submit']):
        return 0    
    print("Error! No Accept or Error")
    return 0


def answer_wrong_question(datapoint):
    query = datapoint['query']
    answer = datapoint['answer']
    prompt = "query: "+"\""+query+"\"\n"
    prompt += "answer: \""+answer+"\"\n"
    prompt += "Generate [Yes] if the response correctly answers the query, and it does not try to adjust the original intent/meaning of the query.\nOtherwise, generate [No]."
    response = llm(prompt)
    #print("prompt:",prompt)
    #print("response:",response)
    print(datapoint['model'],datapoint['id'],'with response:',response)
    if('no' in response or 'No' in response):
        return 0
    if('yes' in response or 'Yes' in response):
        return 1
    print("Error! No yes or no",response)
    return response

def match_score(datapoint,name="Accuracy"):
    return int(datapoint[name+"_x"]==datapoint[name+'_y'])


def normalize_answer(s,normal_method=""):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    def mc_remove(text):
        a1 = re.findall('\([a-zA-Z]\)', text)
        if(len(a1)==0):
            return ""
        return re.findall('\([a-zA-Z]\)', text)[-1]
    if(normal_method=="mc"):
        return mc_remove(s)
    if(type(s)==float):
        return ''
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth, normal_method=""):
    return (normalize_answer(prediction,normal_method=normal_method) == normalize_answer(ground_truth,normal_method=normal_method))

def evaluate(prediction, ground_truth,metric="em"):
    if (metric == "em"):
        return int(exact_match_score(prediction, ground_truth, normal_method=""))