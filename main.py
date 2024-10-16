import argparse
import random
from pipeLLM import pipeAdultData,pipeCreditData,askGPT
from openai import OpenAI
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--k', type=int, default=16)
    parser.add_argument('--dataset', type=str, default="Adult")
    parser.add_argument('--strategy', type=str, default="Balanced_FM_OntimDue")
    parser.add_argument('--llm', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--save_dir_filename', type=str, default='res/tst.txt')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    key = "your API key"
    client = OpenAI(
        # defaults to os.environ.get("OPENAI_API_KEY")
        api_key=key,
    )

    if args.dataset=='Adult':
        intro, few_prompt, xtst = pipeAdultData(args)
        askGPT(args, client,intro, few_prompt, xtst)
    elif args.dataset=='Credit':
        intro, few_prompt, xtst = pipeCreditData(args)
        askGPT(args, client,  intro, few_prompt, xtst)
