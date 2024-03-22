import google.generativeai as genai
from time import sleep
import requests
import base64
import json
import time
import PIL.Image
import os
import pprint
from argparse import ArgumentParser
import traceback
from tqdm import tqdm
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Attention, gemini can receive multiple images at a time

GOOGLE_API_KEY=''
# Or use `os.getenv('GOOGLE_API_KEY')` to fetch an environment variable.


genai.configure(api_key=GOOGLE_API_KEY)

image_path_root = "../images"

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

    
# type: vision
def get_answer_from_gemini_sample(type_, prompt, question, images, model, max_tokens=256, temperature=0.0):
    assert type(prompt) == str or (type(prompt) == list and len(prompt) == 2), "Prompt should be a string or a list of two strings."
    assert question != "" or type(prompt) == str
    
    prompt_input = prompt if question == "" else prompt + question if type(prompt) == str else prompt[0] + question + prompt[1]
    if type_ == "vision":
        
        assert type(images) == str or type(images) == list, "Images should be a string or a list of strings."
        
        if type(images) == str:
            img = PIL.Image.open(os.path.join(image_path_root, 'merged', images))
            response = create_response_gemini([prompt_input, img], model=model, max_tokens=max_tokens, temperature=temperature)
        else:
            ori_paths = [os.path.join(image_path_root, image['path']) for image in images]
            if prompt_input.find(f'<image{len(ori_paths)}>') != -1:
                paths = ori_paths
                assert prompt_input.find(f'<image{len(ori_paths) + 1}>') == -1, "Prompt should not contain <imageN+1> where N is the number of images."
            else:
                # 对paths去重，保持原来的顺序
                paths = list(dict.fromkeys(ori_paths)) 
            
            imgs = [PIL.Image.open(path) for path in paths]
            new_prompt = []
            tmps = prompt_input.split(f'<image')
            for idx, tmp in enumerate(tmps):
                if idx != 0:
                    new_prompt.append(imgs[int(tmp[0])-1])
                    tmp = tmp[2:]

                new_prompt.append(tmp)

            if new_prompt[0] == "":
                new_prompt = new_prompt[1:]
            if new_prompt[-1] == "":
                new_prompt = new_prompt[:-1]

            response = create_response_gemini(new_prompt, model=model, max_tokens=max_tokens, temperature=temperature)
            # import pdb
            # pdb.set_trace()
        
        return_txt = ""
        try:
            return_txt = response.text
        except Exception as e:
            print('error', e)
            traceback.print_exc()
            return_txt = str(response)
        return return_txt
            
        
    


def create_response_gemini(messages, model="gemini-pro-vision", max_tokens=256, temperature=0.0, candidate_count=1, stop_sequences=None):
    safety_settings = [
    {
        "category": "HARM_CATEGORY_DANGEROUS",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_NONE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_NONE",
    },
    ]
    
    model = genai.GenerativeModel(model)
    
    response = model.generate_content(
        messages,
        generation_config=genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
            candidate_count=candidate_count,
            stop_sequences=stop_sequences
        ),
        # safety_settings=safety_settings
    )

    return response




# for debug and test
def test():
    
    type_ = "vision"
    
    prompt = "Here is a diagram of a math problem, please describe the diagram as detailed as possible so that your description can be used to replace the diagram for math problem-solving. \n"
    
    question = ""
    
    image_path = "AMC8_2016_Q22.jpg"
    
    model = 'gemini-1.0-pro-vision-latest'
    
    # for model in genai.list_models():
    #     pprint.pprint(model)
    
    answer = get_answer_from_gemini_sample(type_, prompt, question, image_path, model, max_tokens=512, temperature=0.0)
    
    print(answer)
    print(type(answer))

# for only one merged image
def get_image_path(id_):
    if id_.find('test/geometry/') != -1:
        num = id_.replace('test/geometry/', '').replace('.json', '')
        return os.path.join(image_path_root, 'merged', f'MATH{num}.jpg')
    
    return os.path.join(image_path_root, 'merged', f'{id_}.jpg')

def benchmark_gemini(in_path, save_path):
    
    
    class_prompt = "What branch of mathematics does the problem belong to? \
Choose from the following: 'logic, algebra, counting, visual recognition, topology, probability, statistics, arithmetic, common sense, combinatorics, graph theory, number theory, combinatorial geometry, solid geometry, plane geometry, analytic geometry, descriptive geometry, transformation geometry, others'. \
You are not supposed to solve the problem. Wrap your final answer, a word or a short phrase, in \"\\boxed{}\". \n"
    single_benchmark_prompt = "Answer the question using a single letter or number or word. \n"
    benchmark_prompt = "Please first solve the problem step by step, then put your final answer or a single letter (if it is a multiple choice question) in one \"\\boxed{}\". \n"
    
    prompt = benchmark_prompt
    
    model = 'gemini-1.0-pro-vision-latest'
    
    max_tokens = 4096
    
    questions = load_jsonl(in_path)
    
    last_BEGIN = -1
    retry_num = 0
    max_retry_num = 10
    
    # old_res = load_jsonl('/Users/weikangshi/THU/research/sensetime/Codes/data/benchmark/Gemini/all-20240117-1508_res.jsonl')
    # old_id = [res['extra']['id'] for res in old_res if len(res['extra']['options']) == 0]
    old_id = []
    
    while True:
        try:
            all = load_jsonl(save_path)
        except FileNotFoundError:
            all = []
        
        BEGIN = len(all)
        
        if BEGIN == last_BEGIN:
            retry_num += 1
            print(save_path, f'ERROR: BEGIN == last_BEGIN {last_BEGIN}, retry_num {retry_num}')
            if retry_num > max_retry_num:
                print(save_path, f'ERROR: retry_num > max_retry_num {max_retry_num}, BEGIN {BEGIN}')
                break
        else:
            retry_num = 0
            last_BEGIN = BEGIN
        
        END = len(questions)
        if BEGIN >= END:
            if BEGIN > END:
                print(save_path, f'ERROR: BEGIN {BEGIN} > END {END}')
            else:
                print(save_path, 'DONE', END)
            break
        print(save_path, f'BEGIN: {BEGIN}, END: {END}')
        outs = []

        counter = BEGIN
        
        try:
            for idx, line in enumerate(tqdm(questions[BEGIN:END])):

                if line['id'] not in old_id:
                    question = line['question']
                    options = ''
                    if len(line['options']) > 0:
                        assert len(line['options']) == 5, f"len(line['options']) == {len(line['options'])} != 5"
                        options = f"(A) {line['options'][0]}\n(B) {line['options'][1]}\n(C) {line['options'][2]}\n(D) {line['options'][3]}\n(E) {line['options'][4]}\n"
                    question = f"{question}\n{options}"
                    
                    response = get_answer_from_gemini_sample('vision', prompt, question, line['images'], model, max_tokens=max_tokens, temperature=0.0)

                    res = {'response': response}
        
                    res['system'] = prompt
                    res['model'] = model
                    res['extra'] = line.copy()
                else:
                    res = old_res[counter]
                
                outs.append(res)
                all.append(res)
                counter += 1
                if counter == 3 or counter % 10 == 0 or counter == END:
                    save_jsonl(outs, save_path, mode='a', add_timestamp=False, verbose=False)
                    outs = []
        
        except Exception as e:
            sleep(1)
            print('error', e)
            traceback.print_exc()  # 打印异常信息和堆栈跟踪

        save_jsonl(all, save_path, mode='w', add_timestamp=False, verbose=False)

if __name__ == '__main__':
    parser = ArgumentParser(description="A simple argument parser")
    parser.add_argument("--in_path", type=str, help="input path of data", default='')
    parser.add_argument("--save_path", type=str, help="save path of model outputs", default='')
    args = parser.parse_args()
    
    benchmark_gemini(in_path=args.in_path, save_path=args.save_path)