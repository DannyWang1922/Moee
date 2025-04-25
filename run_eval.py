import subprocess
import sys

model_name = "allenai/OLMoE-1B-7B-0924"
task_types = "STS"
batch_size = "1024"

cmd_list = [
    f"CUDA_VISIBLE_DEVICES=1 python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size}  --emb_info HS --embed_method none",
    f"CUDA_VISIBLE_DEVICES=1 python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size}  --emb_info RW --embed_method none"
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
