from modelservice import make_model
import os, requests, numpy, argparse

def main(): 
    parser = argparse.ArgumentParser(description='LLM API Test.')
    parser.add_argument('--query', type=str, 
                        default="Question: Who is the president of the US? Answer:",
                        help='query') 
    parser.add_argument('--model_name', type=str, 
                        default="claude-1",
                        help='model_name') 
    parser.add_argument('--provider', type=str, 
                        default="anthropic",
                        help='provider')                         
    args = parser.parse_args()
    context = args.query  
    provider = args.provider
    model_name = args.model_name
    print("test of the model service")
    model = make_model(provider, model_name)
    generation = model.getcompletion(context=context,                          
                                     use_save=False,
                             )

    print("---full generation is:")
    print(generation)
    print("---generated alone--")
    print(generation["completion"])
    print("---cost---")
    print(generation['cost'])
    print("--end of generation")
    return 

if __name__ == "__main__":
   main()