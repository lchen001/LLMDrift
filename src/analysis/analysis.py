from .utils import calculate_score, calculate_contain_score, eval_code, direct_usable, answer_wrong_question, match_score
import numpy, pandas

_SCORE_MAP = {
    "Accuracy": calculate_contain_score,
    "Code_Submit":eval_code,
    "Directly Executable":direct_usable,
    "Exact Match":calculate_score,
    "Answer Rate":answer_wrong_question,
}



class Analysis(object):
    def __init__(self,):
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
            if(not(name in data)):
                data[name] = data.apply(method,axis=1)
            scores = data.groupby('model')[name].mean()
            scores_std = data.groupby('model')[name].std(ddof=0) / numpy.sqrt(data.groupby('model').size())
            return scores, scores_std
    
    def get_overlap(self,data,models,name):  
        filtered_df1 = data[data['model']==models[0]]
        filtered_df2 = data[data['model']==models[1]]
        merged_df = pandas.merge(filtered_df1, filtered_df2, on='id')
        merged_df['Answer Overlap'] = merged_df.apply(lambda row: match_score(row, name=name,), axis=1)  
        scores = merged_df['Answer Overlap'].mean()
        
        scores_std = merged_df['Answer Overlap'].std(ddof=0) / numpy.sqrt(merged_df['Answer Overlap'].size)

        return scores, scores_std
        
    def get_code(self,data,name):
        if name in _SCORE_MAP:
            method = _SCORE_MAP[name]
            data[name] = data.apply(method,axis=1)
            #scores = data.groupby('model')[name].mean()
            #scores_std = data.groupby('model')[name].std()
            return #scores, scores_std        
        
