from utils import load_jsonl, save_jsonl
import random

random.seed(0)
input_file = "data/test.jsonl"
examples = load_jsonl(input_file)
output = []

for example in examples:
    if len(example["options"]) == 5:
        reponse = example["options"][random.randint(0, 4)]
    else:
        reponse = random.randint(0, 512)
    output.append({"response": str(reponse), "extra": example})

output_file = "outputs/all-20240122-2245_random_output.jsonl"
save_jsonl(output_file, output, t_stamp=False)

