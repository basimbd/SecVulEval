import re
import json

def get_wrapped_code(full_text, lang="json"):
	pattern = rf'^.*```{lang}(.+)```.*$'
	match = re.search(pattern, full_text, re.DOTALL | re.MULTILINE)
	if match:
		return match.group(1).strip()

def get_json_file_as_dict(filepath: str):
	try:
		with open(filepath, "r") as json_file:
			return json.load(json_file)
	except Exception as err:
		print(err)

def save_dict_as_json(filepath: str, json_dict: dict, indent=None):
	try:
		with open(filepath, "w") as json_file:
			if indent:
				json.dump(json_dict, json_file, indent=indent)
			else:
				json.dump(json_dict, json_file)
	except Exception as err:
		print(err)

def get_json_lines_as_list(filepath: str):
	try:
		with open(filepath, "r") as jsonl_file:
			return [json.loads(line.strip()) for line in jsonl_file.readlines()]
	except Exception as err:
		print(err)

def save_json_lines(filepath: str, json_list: list):
	try:
		with open(filepath, 'w') as jsonl_list:
			for item in json_list:
				# Convert each dictionary to JSON string and write it as a line
				json.dump(item, jsonl_list)
				jsonl_list.write('\n')
	except Exception as err:
		print(err)
