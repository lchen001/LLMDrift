# -*- coding: utf-8 -*-
"""
Created on Mon July  3 21:06:55 2023
@author: Lingjiao Chen
"""
import re, string, subprocess, time

def calculate_score(row):
    # Perform scoring logic based on 'answer' and 'ref_answer'
    # For example, you can use a simple equality check and assign a score of 1 for a match, and 0 for a mismatch
    score = evaluate(row['answer'], row['ref_answer'])
    #print("lens",len(row['answer']),len(row['ref_answer']))
    return score

def calculate_boxed_score(row):
    # Perform scoring logic based on 'answer' and 'ref_answer'
    # For example, you can use a simple equality check and assign a score of 1 for a match, and 0 for a mismatch
    #score = evaluate(row['answer'], row['ref_answer'])
    #print("lens",len(row['answer']),len(row['ref_answer']))
    sub1 = row['ref_answer'].lower()
    full = row['answer'].lower()
    #print("sub1",sub1)
    #print("full",full)

    s1 = find_boxed_substrings(sub1)
    s2 = find_boxed_substrings(full)
    #print("start-- answer",s1,s2)
    if(len(s2)==0):
        score = 0
    elif(s1[0] == s2[0]):
        #print("contained!")
        score = 1
    else:
        score=0
    #print("end--")

    return score    

def calculate_contain_score(row):
    ref1, ref2 = format_answer_yesno(row)
    if(ref1==ref2):
        score = 1
    else:
        score=0
    return score
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
    full1 = full.replace(',','')
    if('['+sub1+']' in add_square_brackets(full1)):
        #print("contained!")
        score = 1
    else:
        print("Wrong with:::",'['+sub1+']',":::answer:::",add_square_brackets(full1))
        
        score=0

    return score

def calculate_contain_mc_score(row):
    # Perform scoring logic based on 'answer' and 'ref_answer'
    # For example, you can use a simple equality check and assign a score of 1 for a match, and 0 for a mismatch
    #score = evaluate(row['answer'], row['ref_answer'])
    #print("lens",len(row['answer']),len(row['ref_answer']))
    sub1 = row['ref_answer'].lower()
    full = row['answer'].lower()
    sub2 = "the answer is "+sub1
    if(sub2 in full):
        score = 1
    else:
        #print(sub2, "text::",full)
        score=0

    return score

def calculate_contain_mc_zshot_score(row):
    # Perform scoring logic based on 'answer' and 'ref_answer'
    # For example, you can use a simple equality check and assign a score of 1 for a match, and 0 for a mismatch
    #score = evaluate(row['answer'], row['ref_answer'])
    #print("lens",len(row['answer']),len(row['ref_answer']))
    sub1 = row['ref_answer'].lower()
    if(len(sub1)>=3):
        sub2 = sub1[1]
    else:
        sub2 = sub1
    full = row['answer'].lower()
    if(sub1 in full):
        score = 1
    elif( sub2 in full ):
        score = 1
    else:
        print("Wrong answer:", sub1, "text::",full)
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

def clean_eval_code(datapoint):
    def _clean(input_string):
        pattern = re.compile(r'```python(.*?)```', re.DOTALL)
        result = pattern.search(input_string)
        if result:
            extracted_string = result.group(1)
            return extracted_string
        else:
            print("No match found")
            return input_string
    #time.sleep(1)
    # get filename
    title = datapoint['id'].split("-", 1)[1] 
    # get code
    code = _clean(datapoint['answer'])
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

def executable(datapoint):
    if ('Accepted' in datapoint['Clean_Code_Submit']):
        return 1
    if ('Error' in datapoint['Clean_Code_Submit']):
        return 0
    if ('Wrong' in datapoint['Clean_Code_Submit']):
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
    try:
        return int(datapoint["format_answer_x"]==datapoint['format_answer_y'])
    except:
        #print("Use original score")
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


def find_boxed_substrings(input_string):
    pattern = r"boxed{([^}]*)}"  # Regular expression pattern to match bbox{...}
    matches = re.findall(pattern, input_string)
    return matches        


def answer_survey_mc_score(datapoint):
    pattern = r'\([A-Za-z]\)\. Refused'
    # Test the pattern on a string
    string = str(datapoint['answer'])

    match = re.search(pattern, string)

    pattern2 = r'\([A-Za-z]\)'
    match2 = re.match(pattern2, string)
    pattern3 = r'\([A-Za-z]\)'
    match3 = re.search(pattern3, string)
    if (match):
        #print("Refused with:",string)
        score = 0
    elif(match2):
        #print("match 2")
        score = 1
    elif(match3):
        #print("match3")
        score = 1
    else:
        #print("No match found.",string)
        score = 0
    return score

def string_follow_score(datapoint):
    return int(datapoint['format_answer']==1.0)

def diversity_score(datapoints):
    return

def format_answer_survey(datapoint):
    format1 = format_mc(datapoint['ref_answer'])
    format2 = format_mc(datapoint['answer'])
    return format1, format2


def format_mc(string):
    pattern = r'\([A-Za-z]\)'
    match = re.search(pattern, string)    
    if(match):
        return match[0]
    return string

def format_answer_raw(datapoint):
    return datapoint['ref_answer'], datapoint['answer']

def format_answer_yesno(datapoint):
    def add_square_brackets(string):
        string = string.replace(","," ")
        string = string.replace("."," ")
        words = string.split()
        words_with_brackets = [f"[{word}]" for word in words]
        result = " ".join(words_with_brackets)
        return result
    def _get_yes_no(string):
        string_low = string.lower()
        #string_low = string_low.replace(",","")
        pattern = r'\[yes\]'
        match = re.search(pattern, string_low) 
        if(match):
            return 'yes'
        pattern = r'\[no\]'
        match = re.search(pattern, string_low) 
        if(match):
            return 'no'
        #return string

        return 'undetermined'

    full1 = datapoint['answer'].replace(',','')
    #full1 = datapoint['answer']

    ref1 = _get_yes_no(add_square_brackets(datapoint['ref_answer']))
    ref2 = _get_yes_no(add_square_brackets(datapoint['answer']))
    #ref1 = _get_yes_no(datapoint['ref_answer']
    return ref1, ref2


def format_boxed(datapoint):
    def _get_box(string):
        string_low = string.lower()
        string_low = string_low.replace(".","")
        pattern = r'boxed\{[0-9]\}'
        match = re.search(pattern, string_low) 
        if(match):
            return match[0]
        else:
            return 'undetermined'
            return string_low

    ref1 = _get_box(str(datapoint['ref_answer']))
    ref2 = _get_box(str(datapoint['answer']))

    return ref1, ref2


def format_answer_em(datapoint):
    normal_method='em'
    ref1 = normalize_answer(datapoint['ref_answer'],normal_method=normal_method)
    ref2 = normalize_answer(datapoint['answer'],normal_method=normal_method)
    return ref1, ref2

def format_stringfollow(datapoint):
    temp= datapoint['ref_answer']
    ans1 = stringfollow(datapoint)
    return temp, ans1

def stringfollow(datapoint):
    string = extract_words(datapoint['answer'].lower())
    text = datapoint['query']
    letter = extract_letter(text)
    start = extract_start(text)
    # start 
    if(start):
        matches = [item[0]==letter for item in string]
    # or end
    else:
        matches = [item[-1]==letter for item in string]
    return sum(matches)/len(matches)


def extract_letter(text):
    pattern = r'with\s+"(\w)"'
    match = re.search(pattern, text)
    if match:
        return match.group(1)
    else:
        return None
    
def extract_start(text):
    pattern = r'start'
    match = re.search(pattern, text)
    if match:
        return match.group(0)
    else:
        return None  
def extract_words(text):
    pattern = r'\b\w+\b'
    return re.findall(pattern, text)


def latency_score(datapoint):
    return datapoint['latency']