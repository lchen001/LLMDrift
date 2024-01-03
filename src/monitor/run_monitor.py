#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  9 15:45:06 2023

@author: lingjiao
"""

import argparse, json, os
from llmmonitor import LLMMonitor, configparse

def monitor(query_params,
             model_list,
             model_params,
             trail,
             system_prompts,
             date_eval):
    
    MyLLMMonitor = LLMMonitor()
    MyLLMMonitor.evaluate(query_params,
                 model_list,
                 model_params,
                 trail,
                 system_prompts,
                 str(date_eval))
    return

def main(args):
    print('Hello, ' + args.config)
    print("API Key",os.environ['OPENAI_API_KEY'])
    config = json.load(open(args.config))
    query_params, model_list, model_params, trail, system_prompts = configparse(config)
    monitor(query_params,
             model_list,
             model_params,
             trail,
             system_prompts=system_prompts,
             date_eval=args.date)
    return 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--config', type=str, default='World',
                        help='config path of the dataset')
    parser.add_argument('--date', type=str, default='20230702',
                        help='monitoring date')

    args = parser.parse_args()
    main(args)
