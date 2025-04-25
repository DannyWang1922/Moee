import mteb
import sys
import logging
from prettytable import PrettyTable
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import numpy as np

# 设置日志
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

# 选择设备
device = torch.device("cuda:0" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))

# 加载模型
model = mteb.get_model("BAAI/bge-base-en-v1.5")

# 选择任务
task_list = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16', 'STSBenchmark', "SICK-R"]
# task_list = ['STS12']
tasks = mteb.get_tasks(tasks=task_list)

# 运行评估
evaluation = mteb.MTEB(tasks=tasks)
results = evaluation.run(model, overwrite_results=True)

task_names = ['Model'] + task_list 
task_scores = []
for task, _ in enumerate(task_list):
    score = results[task].scores["test"][0]["main_score"] * 100 
    task_scores.append(round(score, 2))
average_score = round(np.mean(task_scores), 2)

# 打印表格
def print_table(model_name, task_names, scores, avg_score):
    tb = PrettyTable()
    tb.field_names = ["Model"] + task_names + ["Avg."]
    tb.add_row([model_name] + scores + [avg_score])
    print(tb)

print_table("BAAI/bge-base-en-v1.5", task_list, task_scores, average_score)




