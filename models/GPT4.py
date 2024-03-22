import time
import os
import json
import openai
from multiprocessing import Pool
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))


client = openai.OpenAI(api_key = "your_api_key_here")

MODEL = 'gpt-4'
if not os.path.exists(f'{MODEL}_answer.jsonl'):
    with open(f'{MODEL}_answer.jsonl', 'w') as f:
        pass

def load_jsonl(file_path)->list:
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

# Define the function to process each example
def geninput(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    # input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    return input


def ask_gpt4(question):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": question
            }
        ],
        model=MODEL,
        max_tokens=2048
    )
    return chat_completion.model_dump()

def process_example(example):
    question = geninput(example)
    for i in range(10):
        question = question.replace(f'<image{i}>', '').strip()
    response = ask_gpt4(question)
    response['input'] = question
    response['extra'] = example
    # print(json.dumps(response, ensure_ascii=False))
    return response

def test(path, output, processor, reset=False):
    if reset or not os.path.exists(output):
        with open(output, 'w') as f:
            pass
    with open(output, 'r') as f:
        processed_num = len(f.readlines())
    data = load_jsonl(path)[processed_num:]
    for example in tqdm(data):
        answer = processor(example)
        with open(output, 'a') as f:
            f.write(json.dumps(answer, ensure_ascii=False)+'\n')
        assert answer['choices'][0]['finish_reason'] in ['stop', 'length']



if __name__ == '__main__':
    input_path = "./data/test.jsonl"
    output_path = f"{MODEL}_answer.jsonl"
    test(input_path, output_path, process_example)