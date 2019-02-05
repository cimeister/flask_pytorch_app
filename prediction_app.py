from flask import Flask
app = Flask(__name__)

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForRegression
import torch
import json
import os

import argparse

@app.route('/')
def hello_world():
    return 'Hello, World!'

parser = argparse.ArgumentParser()

## Required parameters
parser.add_argument("--model_dir",
                    default=None,
                    type=str,
                    required=True,
                    help="The model directory. Should contain the config json and pytorch model bin file.")
args = parser.parse_args()

if not os.path.exists(args.model_dir):
    raise ValueError("Model directory ({}) does not exist.".format(args.model_dir))

config_file = os.path.join(args.model_dir, "config.json")
with open(config_file) as f:
    config = json.load(f)

device = torch.device('cpu')

model_state_dict = torch.load(os.path.join(args.model_dir, "pytorch_model.bin"))
model = BertForRegression.from_pretrained(config['bert_model'], state_dict=model_state_dict, 
        inner_layer_size=config['layer1'], outer_layer_size=config['layer2'])

tokenizer = BertTokenizer.from_pretrained(config['bert_model'], do_lower_case=config['lowercase'])
model.to(device)
model.eval()

@app.route('/predict/<sen1>/<sen2>', methods=['GET'])
def predict(sen1, sen2):
	sen1, sen2 = sen1.replace('_', ' ').lower(), sen2.replace('_', ' ').lower()
	input_id, segment_id, input_mask = tokenize(sen1, sen2)

	id_tensor = torch.tensor([input_id], dtype=torch.long)
	segment_tensor = torch.tensor([segment_id], dtype=torch.long)
	mask_tensor = torch.tensor([input_mask], dtype=torch.long)
	
	with torch.no_grad():
		y = model(id_tensor, segment_tensor, mask_tensor)
	return str(y.numpy()[0][0]) + '\n'

def tokenize(sen1, sen2):
	tokens_a = tokenizer.tokenize(sen1)
	tokens_b= tokenizer.tokenize(sen2)
	tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
	segment_ids = [0] * len(tokens)

	tokens += tokens_b + ["[SEP]"]
	segment_ids += [1] * (len(tokens_b) + 1)

	input_ids = tokenizer.convert_tokens_to_ids(tokens)

	# The mask has 1 for real tokens and 0 for padding tokens. Only real
	# tokens are attended to.
	input_mask = [1] * len(input_ids)

	# Zero-pad up to the sequence length.
	padding = [0] * (config['max_seq_length'] - len(input_ids))
	input_ids += padding
	input_mask += padding
	segment_ids += padding
	return input_ids, segment_ids, input_mask


if __name__ == "__main__":
	app.run()

