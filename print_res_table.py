import os
import json
from pyexpat import model
from prettytable import PrettyTable


def print_table(model_name, task_names, scores, avg_score):
    tb = PrettyTable()
    tb.field_names = ["Model"] + task_names + ["Avg."]
    tb.add_row([model_name] + scores + [avg_score])
    print(tb)

def read_score_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
        try:
            return data["scores"]["test"][0]["main_score"] * 100
        except (KeyError, IndexError, TypeError):
            return 0 

def collect_scores(method_prefix):
    scores = []
    for task in task_list:
        task_folder_name = f"{method_prefix}_{task}"
        json_file_name = f"{task}.json"
        json_path = os.path.join(base_path, task_folder_name, sub_path, json_file_name)

        if os.path.exists(json_path):
            score = read_score_from_json(json_path)
        else:
            score = 0
            print(f"File not found: {json_path}")
        scores.append(round(score, 2))
    
    avg_score = round(sum([s for s in scores]) / len(scores), 2)
    return scores, avg_score


model_name = "OLMoE-1B-7B-0924"
task_list = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', "SICK-R"]
task_types = "STS"

methods = ["HS", "RW", "MoEE"]
base_path = f"mteb_results_ablation/{model_name}/{task_types}"
sub_path  = "no_model_name_available/no_revision_available"

for method in methods:
    task_scores, average_score = collect_scores(method)
    print_table(method, task_list, task_scores, average_score)
