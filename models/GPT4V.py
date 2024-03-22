import base64
import openai
import requests
import json
from tqdm import tqdm
import os
from multiprocessing import Pool
from tqdm import tqdm
import random
from time import sleep
os.chdir(os.path.dirname(os.path.abspath(__file__)))


# OpenAI API Key
client = openai.OpenAI(api_key = "your_api_key_here")





# Define the function to process each example
def geninput(example):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    input = f"{question}\n{options}\nAnswer the question using a single word or phrase."
    return input

# Function to encode the image
def encode_image(image_path):
    if image_path.startswith("http"):
        return image_path
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    return f"data:image/jpeg;base64,{base64_image}"

def load_jsonl(file_path)->list:
    with open(file_path, 'r') as f:
        return [json.loads(line.strip()) for line in f]



def ask_gpt4v(question, image_paths):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    content = [{"type": "text", "text": question}]
    for image_path in image_paths:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": encode_image(image_path)
            }
        })

    payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "max_tokens": 4096
    }

    # print(json.dumps(payload, indent=4))
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload).json()

    for item in content:
        if item['type'] == 'image_url':
            item['image_url']['url'] = image_path
    
    response['user'] = content
    return response



# Define the function to process each example
def process_example(example, img_folder="./images"):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    
    input = 'Please solve the problem and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    
    
    image_paths = []
    assert len(example['images']) == input.count('<image'), example
    for idx, img in enumerate(example['images']):
        img = img['path']
        idx += 1
        new_path = f"working/gpt4v/<image{idx}>.jpg"
        os.system(f'cp "{os.path.join(img_folder, img)}" "{os.path.join(img_folder, new_path)}"')
        image_paths.append(os.path.join(img_folder, new_path))
           
    
    answer = ask_gpt4v(input, image_paths)
    answer['extra'] = example
    return answer

# Define the function to process each example
def process_example_mergedimg(example, img_folder="./images"):
    question = example['question']
    options = ''
    if len(example['options']) > 0:
        assert len(example['options']) == 5, example
        if ''.join(example['options']) != 'ABCDE':
            options = f"(A) {example['options'][0]}\n(B) {example['options'][1]}\n(C) {example['options'][2]}\n(D) {example['options'][3]}\n(E) {example['options'][4]}\n"
    
    input = 'Please solve the problem and put your answer in one "\\boxed{}". If it is a multiple choice question, only one letter is allowed in the "\\boxed{}".\n'+f"{question}\n{options}"
    
    assert len(example['images']) == input.count('<image'), example
    img = id2mergedpath(example['id'])
    new_path = f"working/gpt4v/input.jpg"
    os.system(f'cp "{os.path.join(img_folder, img)}" "{os.path.join(img_folder, new_path)}"')
    image_paths = [os.path.join(img_folder, new_path)]
           
    
    answer = ask_gpt4v(input, image_paths)
    answer['extra'] = example
    return answer





def testp(path, output, processor, reset=False):
    if reset or not os.path.exists(output):
        with open(output, 'w') as f:
            pass
    with open(output, 'r') as f:
        processed_num = len(f.readlines())
    data = load_jsonl(path)[processed_num:]

    # Create a multiprocessing pool with the number of available CPUs
    pool = Pool(processes=4)

    # Map the helper function to the data using the pool
    results = []
    for i, answer in enumerate(tqdm(pool.imap(processor, data), total=len(data))):
        results.append(answer)
        if (i + 1) % 10 == 0:
            with open(output, 'a') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False)+'\n')
                results = []

    # Write the remaining results to the output file
    with open(output, 'a') as f:
        for answer in results:
            f.write(json.dumps(answer, ensure_ascii=False)+'\n')

def test(path, output, processor, reset=False):
    if reset or not os.path.exists(output):
        with open(output, 'w') as f:
            pass
    with open(output, 'r') as f:
        processed_num = len(f.readlines())
    data = load_jsonl(path)[processed_num:]
    for example in tqdm(data):
        answer = processor(example)
        while "error" in answer:
            if "Rate limit reached for gpt-4-vision-preview in organization" in answer["error"]["message"]:
                print(answer["error"]["message"])
                sleep(60)
                answer = processor(example)
            elif answer["error"]["message"] == "Your input image may contain content that is not allowed by our safety system.":
                break
            else:
                print(json.dumps(answer, ensure_ascii=False))
                exit(-1)
        with open(output, 'a') as f:
            f.write(json.dumps(answer, ensure_ascii=False)+'\n')
    

if __name__ == '__main__':
    path = "./inputs/test.jsonl"
    test(path,  path.replace('.jsonl', '_gpt4v_output.jsonl'), processor=process_example_mergedimg)