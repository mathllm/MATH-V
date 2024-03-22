# For prerequisites running the following sample, visit https://help.aliyun.com/document_detail/611472.html
from http import HTTPStatus
import dashscope
import os
from time import sleep
import json
from tqdm import tqdm
from multiprocessing import Pool
import traceback
os.chdir(os.path.dirname(os.path.abspath(__file__)))

dashscope.api_key = ''

def id2mergedpath(id):
    return f"merged/{id}.jpg".replace('test/geometry/', 'MATH').replace('.json', '')


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


def load_jsonl(file_path)->list:
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]

def ask_qwvl(question, image_paths):
    content = []
    for image_path in image_paths:
        content.append({
            "image": f"file://{image_path}"
        })
    content.append({"text": question})

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    # print(json.dumps(messages, indent=4))
    response = dashscope.MultiModalConversation.call(model='qwen-vl-max',
                                                     messages=messages)
    # The response status_code is HTTPStatus.OK indicate success,
    # otherwise indicate request is failed, you can get error code
    # and message from code and message.
    if response.status_code == HTTPStatus.OK:
        pass
        # print(response)
    else:
        print(response.code)  # The error code.
        print(response.message)  # The error message.
        # exit(-1)
        
    response["input"] = messages
    return response

# Define the function to process each example
def process_example(example, img_folder="./images"):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    
    # input = 'Please solve the problem step by step and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    
    
    image_paths = []
    assert len(example['images']) == input.count('<image'), example
    for idx, img in enumerate(example['images']):
        img = img['path']
        idx += 1
        new_path = f"working/qwenvl/<image{idx}>.jpg"
        os.system(f'cp "{os.path.join(img_folder, img)}" "{os.path.join(img_folder, new_path)}"')
        image_paths.append(os.path.join(img_folder, new_path))
           
    
    answer = ask_qwvl(input, image_paths)
    answer['extra'] = example
    return answer

# Define the function to process each example
def process_example_merged_img(example, img_folder="./images"):
    input = geninput(example)

    image_paths = []
    img = id2mergedpath(example['id'])
    new_path = f"working/qwenvl/<image>.jpg"
    os.system(f'cp "{os.path.join(img_folder, img)}" "{os.path.join(img_folder, new_path)}"')
    image_paths.append(os.path.join(img_folder, new_path))
           
    
    answer = ask_qwvl(input, image_paths)
    answer['extra'] = example
    return answer

def test(path, output, processor, reset=False):
    if reset or not os.path.exists(output):
        with open(output, 'w') as f:
            pass
    with open(output, 'r') as f:
        processed_num = len(f.readlines())
    data = load_jsonl(path)[processed_num:]
    for example in tqdm(data):
        answer = processor(example)
        while answer["status_code"] != HTTPStatus.OK:
            if answer["message"] == "Requests rate limit exceeded, please try again later.":
                sleep(21)
                answer = processor(example)
            else:
                exit(-1)
        with open(output, 'a') as f:
            f.write(json.dumps(answer, ensure_ascii=False)+'\n')



if __name__ == '__main__':
    path = "./inputs/test.jsonl"
    test(path,  path.replace('.jsonl', '_qwenvlmax_cot_mergedimg_output.jsonl'), process_example_merged_img)