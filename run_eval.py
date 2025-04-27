import subprocess
import sys

model_name = "allenai/OLMoE-1B-7B-0924"
task_types = "STS"
batch_size = "1024"

cmd_list = [
    f"CUDA_VISIBLE_DEVICES=1 python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method none --emb_info HS ",
    f"CUDA_VISIBLE_DEVICES=1 python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method none --emb_info RW ",
    f"CUDA_VISIBLE_DEVICES=1 python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method none --emb_info MoEE ",
    f"CUDA_VISIBLE_DEVICES=1 python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method prompteol --emb_info HS ",
    f"CUDA_VISIBLE_DEVICES=1 python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method prompteol --emb_info RW ",
    f"CUDA_VISIBLE_DEVICES=1 python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method prompteol --emb_info MoEE "
    ]

for cmd in cmd_list:
    print(f"\nRunning cmd: {cmd}\n")
    try:
        # subprocess.run(cmd, shell=True, check=True, stdout=sys.stdout, stderr=sys.stderr)
        print(f"Command Done: {cmd}")
        print("=" * 100)
    except subprocess.CalledProcessError as e:
        print(f"Command execution failure: {e}")
        print("Error message:")
        print(e.stderr)
