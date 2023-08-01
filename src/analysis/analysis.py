from .utils import calculate_score, calculate_contain_score, eval_code, direct_usable, answer_wrong_question, match_score, calculate_contain_mc_score, calculate_boxed_score, clean_eval_code, executable, calculate_contain_mc_zshot_score
from .utils import answer_survey_mc_score
from .utils import format_answer_survey, format_answer_raw, format_answer_yesno, format_boxed, format_answer_em

import numpy, pandas

_SCORE_MAP = {
    "Accuracy": calculate_contain_score,
    "Code_Submit":eval_code,
    "Clean_Code_Submit":clean_eval_code,
    "Executable":executable,
    "Directly Executable":direct_usable,
    "Exact Match":calculate_score,
    "Answer Rate":answer_wrong_question,
    "Multiple-choice Accuracy":calculate_contain_mc_score,
    "Zeroshot-Multiple-choice Accuracy":calculate_contain_mc_zshot_score,
    "Math Accuracy":calculate_boxed_score,
    "Survey Rate":answer_survey_mc_score,
}

_FORMAT_MAP = {
    "Survey Rate":format_answer_survey,
    "Exact Match":format_answer_raw,
    "Multiple-choice Accuracy":format_answer_survey,
    "Accuracy":format_answer_yesno,
    "Math Accuracy":format_boxed,
}


_NAME_MAP = {
    "Accuracy": "Accuracy",
    "Code_Submit":"Code_Submit",
    "Clean_Code_Submit":"Clean_Code_Submit",
    "Executable":"Executable",
    "Directly Executable":"Directly EXE",
    "Exact Match":"Exact Match",
    "Answer Rate":"Response Rate",
    "Multiple-choice Accuracy":"Accuracy",
    "Math Accuracy":"Accuracy",
    "Zeroshot-Multiple-choice Accuracy":"Accuracy",
    "Survey Rate":"Response Rate",
}

class Analysis(object):
    def __init__(self,):
        return

    def format_answer(self,data,name):
        if name in _FORMAT_MAP:
            method = _FORMAT_MAP[name]
            if(name in _NAME_MAP):
                name = _NAME_MAP[name]
            if(not(name in data)):
                data[['format_ref_answer','format_answer']] = data.apply(lambda x: pandas.Series(method(x)), axis=1)
            return data        
        return
            
    def get_verbosity(self,data):
        data['answer'] = data['answer'].astype(str)
        data['verbosity'] = data['answer'].apply(len)
        average_lengths = data.groupby('model')['verbosity'].mean()
        scores_std = data.groupby('model')['verbosity'].std(ddof=0) / numpy.sqrt(data.groupby('model').size())
        return average_lengths, scores_std

    def get_score(self,data,name):
        if name in _SCORE_MAP:
            method = _SCORE_MAP[name]
            if(name in _NAME_MAP):
                name = _NAME_MAP[name]
            if(not(name in data)):
                data[name] = data.apply(method,axis=1)
            scores = data.groupby('model')[name].mean()
            scores_std = data.groupby('model')[name].std(ddof=0) / numpy.sqrt(data.groupby('model').size())
            return scores, scores_std
    
    def get_overlap(self,data,models,name):  
        filtered_df1 = data[data['model']==models[0]]
        filtered_df2 = data[data['model']==models[1]]
        merged_df = pandas.merge(filtered_df1, filtered_df2, on='id')
        if(name in _NAME_MAP):
                name = _NAME_MAP[name]
        merged_df['Answer Overlap'] = merged_df.apply(lambda row: match_score(row, name=name,), axis=1)  
        scores = merged_df['Answer Overlap'].mean()
        scores_std = merged_df['Answer Overlap'].std(ddof=0) / numpy.sqrt(merged_df['Answer Overlap'].size)

        return scores, scores_std

    def get_mismatch(self,data,models,name):  
        filtered_df1 = data[data['model']==models[0]]
        filtered_df2 = data[data['model']==models[1]]
        merged_df = pandas.merge(filtered_df1, filtered_df2, on='id')
        if(name in _NAME_MAP):
                name = _NAME_MAP[name]
        merged_df['Answer Mismatch'] = merged_df.apply(lambda row: 1-match_score(row, name=name,), axis=1)  
        scores = merged_df['Answer Mismatch'].mean()
        scores_std = merged_df['Answer Mismatch'].std(ddof=0) / numpy.sqrt(merged_df['Answer Mismatch'].size)

        return scores, scores_std        
        
    def get_code(self,data,name):
        if name in _SCORE_MAP:
            method = _SCORE_MAP[name]
            data[name] = data.apply(method,axis=1)
            return         
        
