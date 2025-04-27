import subprocess
import sys

model_name = "allenai/OLMoE-1B-7B-0924" # allenai/OLMoE-1B-7B-0924, Qwen/Qwen1.5-MoE-A2.7B
task_types = "Classification,Clustering,PairClassification,Reranking,STS,Summarization"
batch_size = "16"
nv_command = "NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 CUDA_VISIBLE_DEVICES=0 "

cmd_list = [
    nv_command +f"python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method none --emb_info HS ",
    nv_command +f"python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method none --emb_info RW ",
    nv_command +f"python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method none --emb_info MoEE ",
    nv_command +f"python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method prompteol --emb_info HS ",
    nv_command +f"python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method prompteol --emb_info RW ",
    nv_command +f"python eval_mteb.py --base_model {model_name} --use_4bit --task_types {task_types} --batch_size {batch_size} --embed_method prompteol --emb_info MoEE "
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
