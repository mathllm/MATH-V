from openai import OpenAI
import openai
from argparse import ArgumentParser

import requests
import base64
import json
import time
import os
import traceback
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# OpenAI Key
api_key = ""

client = OpenAI(api_key=api_key)

image_path_root = "../images"

def download_img(idx, file_id, save_path):
    image_data = client.files.content(file_id)
    image_data_bytes = image_data.read()
    
    with open(os.path.join(save_path, f'{idx}_{file_id}.png'), 'wb') as file:
        file.write(image_data_bytes)
    
def timestamp() -> str:
    nowtime = time.strftime('-%Y%m%d-%H%M', time.localtime(time.time()))
    print(nowtime)  
    return nowtime  

def save_jsonl(data: list, path: str, mode='w', add_timestamp=True, verbose=True) -> None:
    if add_timestamp:
        file_name = f"{path.replace('.jsonl','')}{timestamp()}.jsonl"
    else:
        file_name = path
    with open(file_name, mode, encoding='utf-8') as f:
        if verbose:
            for line in tqdm(data, desc='save'):
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        else:
            for line in data:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')

def load_jsonl(file_name):
    with open(file_name, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]
    return data

# Function to encode the image
def encode_image(image_path):
    image_path = os.path.join(image_path_root, image_path)
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
# type: vision, code
def get_answer_from_gpt_sample(idx, type_, instructions, prompt, question, image_path, model, max_tokens=256, temperature=0.0):
    assert type(prompt) == str or (type(prompt) == list and len(prompt) == 2), "Prompt should be a string or a list of two strings."
    assert question != "" or type(prompt) == str
    
    prompt_input = prompt if question == "" else prompt + question if type(prompt) == str else prompt[0] + question + prompt[1]
    
    # 暂时只写了单张图片，其实可以多张图片
    if type_ == "vision":
        assert image_path != "", "Vision model requires image input."

        response = create_response_4V(messages=[
                    {"role": "user", 
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_input
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/{image_path.split('.')[-1]};base64,{encode_image(image_path)}"
                            }
                        }
                    ]
                    }
                ], model=model, max_tokens=max_tokens, temperature=temperature)

        # 此处response是一句话
        # response = response['choices'][0]['message']['content']
    
    elif type_ == 'text':
        assert image_path == "", "Text model currently does not support image input."
        
        response = create_response_4(messages=[
            {"role": "user", "content": prompt_input}
        ], model=model, max_tokens=max_tokens, temperature=temperature)
    
    return response
    


def create_response_4V(messages, model="gpt-4-vision-preview", max_tokens=256, temperature=0.0):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    request_url = "https://api.openai.com/v1/chat/completions"
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    response = requests.post(request_url, headers=headers, json=payload)
    return response.json()

def create_response_4(messages, model="gpt-4-turbo-preview", max_tokens=256, temperature=0.0):

    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature
    )
    
    return response.model_dump()

# for debug and test
def test():
    
    type_ = 'text'
    
    instructions = 'You are a personal math tutor. Write and run code to answer math questions.'
    
    prompt = ""
    
    question = "For trapezoid J K L M, A and B are midpoints of the legs. If A B = 57 and K L = 21, find J M.\nChoices:\nA: 21\nB: 57\nC: 87\nD: 93"
    
    image_path = ""
    
    model = 'gpt-4-turbo-preview'
    
    answer = get_answer_from_gpt_sample(-1, type_, instructions, prompt, question, image_path, model, max_tokens=4096, temperature=0.0)
    print(answer)
    print(type(answer))
    
    json.dump(answer, open("answer.json", "w", encoding="utf-8"), ensure_ascii=False, indent=4)

# for only one merged image
def get_image_path(id_):
    if id_.find('test/geometry/') != -1:
        num = id_.replace('test/geometry/', '').replace('.json', '')
        return os.path.join(image_path_root, 'merged', f'MATH{num}.jpg')
    
    return os.path.join(image_path_root, 'merged', f'{id_}.jpg')

def benchmark_4V(v_model, in_path, save_path):
    
    v_prompt = "Here is a diagram of a math problem, please describe the diagram as detailed as possible so that your description can be used to replace the diagram for math problem-solving. \n"
    
    max_tokens = 4096
    
    questions = load_jsonl(in_path)
    old_id = []
    
    while True:
        try:
            all = load_jsonl(save_path)
        except FileNotFoundError:
            all = []
        
        BEGIN = len(all)
        END = len(questions)
        if BEGIN >= END:
            if BEGIN > END:
                print(save_path, 'ERROR: BEGIN > END')
            else:
                print(save_path, 'DONE', END)
            break
        print(save_path, f'BEGIN: {BEGIN}, END: {END}')
        outs = []

        counter = BEGIN
        
        try:
            for idx, line in enumerate(tqdm(questions[BEGIN:END])):
                if line['id'] not in old_id:
                    image_path = get_image_path(line['id'])

                    v_response = get_answer_from_gpt_sample(counter, 'vision', '', v_prompt, "", image_path, v_model, max_tokens=max_tokens, temperature=0.0)


                    if 'error' in v_response and 'message' in v_response['error']:
                        print(v_response['error']['message'])
                        if 'The server had an error processing your request.' in v_response['error']['message']:
                            sleep_time = 30
                            print(f'sleep {sleep_time * 2}s')
                            time.sleep(sleep_time * 2)
                            break
                        if 'Please try again in ' in v_response['error']['message']:
                            sleep_time = float(v_response['error']['message'].split('Please try again in ')[1].split('s.')[0])
                            print(f'sleep {sleep_time * 2}s')
                            time.sleep(sleep_time * 2)
                            break
                        print('get image description failed, skip', line['id'])
                    else:
                        image_decs = v_response['choices'][0]['message']['content']

                    res = {
                        "image_decs": {
                            "prompt": v_prompt,
                            "content": image_decs,
                            "response": v_response
                        },
                        "extra": line
                    }
                else:
                    res = old_res[counter]
                
                outs.append(res)
                all.append(res)
                counter += 1
                if counter == 3 or counter % 10 == 0 or counter == END:
                    save_jsonl(outs, save_path, mode='a', add_timestamp=False, verbose=False)
                    outs = []
                
        
        except Exception as e:

            print('error', e)
            traceback.print_exc()  # 打印异常信息和堆栈跟踪

        save_jsonl(all, save_path, mode='w', add_timestamp=False, verbose=False)

def benchmark_text(text_model, in_path, save_path, old_path):
    
    max_tokens = 4096
    if '3.5' in text_model:
        max_tokens = 2048
    
    questions = load_jsonl(in_path)
    
    try:
        old_res = load_jsonl(old_path)
        old_id = [res['extra']['id'] for res in old_res if res['response'] != None]
    except:
        old_id = []
    print(f'{old_id=}')
    
    images_decs = load_jsonl('./all-20240124-1958_gpt-v.jsonl')
    
    while True:
        try:
            all = load_jsonl(save_path)
        except FileNotFoundError:
            all = []
        
        BEGIN = len(all)
        END = len(questions)
        if BEGIN >= END:
            if BEGIN > END:
                print(save_path, 'ERROR: BEGIN > END')
            else:
                print(save_path, 'DONE', END)
            break
        print(save_path, f'BEGIN: {BEGIN}, END: {END}')
        outs = []

        counter = BEGIN
        
        try:
            for idx, line in enumerate(tqdm(questions[BEGIN:END])):
                if line['id'] not in old_id:

                    v_prompt = images_decs[counter]['image_decs']['prompt']
                    image_decs = images_decs[counter]['image_decs']['content']
                    v_response = images_decs[counter]['image_decs']['response']

                    if image_decs != '':
                        text_prompt = "Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\". \n \
Here is the natural description of the figure, please solve the following problem based on the description: \n" + image_decs + \
"\n\nThe problem is: \n"
# "\n\nThe problem is: \n", "\n\nRemember to put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\"."]

                        response = get_answer_from_gpt_sample(counter, 'text', '', text_prompt, line['question'], "", text_model, max_tokens=max_tokens, temperature=0.0)
                    else:
                        print('image description empty', line['id'])
                        response = None

                    res = {
                        "image_decs": {
                            "prompt": v_prompt,
                            "content": image_decs,
                            "response": v_response
                        },
                        'prompt': text_prompt,
                        "response": response,
                        "model": text_model, 
                        "extra": line
                    }
                else:
                    res = old_res[counter]
                
                outs.append(res)
                all.append(res)
                counter += 1
                if counter == 3 or counter % 10 == 0 or counter == END:
                    save_jsonl(outs, save_path, mode='a', add_timestamp=False, verbose=False)
                    outs = []
                
        
        except Exception as e:

            print('error', e)
            traceback.print_exc()  # 打印异常信息和堆栈跟踪

        save_jsonl(all, save_path, mode='w', add_timestamp=False, verbose=False)

if __name__ == '__main__':
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("--model", type=str, help="model name", default='')
    parser.add_argument("--in_path", type=str, help="input path of data", default='')
    parser.add_argument("--save_path", type=str, help="save path of model outputs", default='')
    parser.add_argument("--old_path", type=str, help="old save path", default='')
    args = parser.parse_args()
    
    if 'vision' in args.model:
        benchmark_4V(args.model, args.in_path, args.out_path)
    else:
        benchmark_text(args.model, args.in_path, args.out_path, args.old_path)