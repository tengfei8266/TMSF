CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "RFAN"
wait
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "SRCNN"
wait
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "RDN"
wait
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "EDSR"
wait
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "RCAN"
wait
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "DATN"
wait
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "HNCT"
wait
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "HSTN"
wait
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "ARBRCAN"
wait
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 python -m torch.distributed.run --nproc_per_node=6 main.py --model_name "ARBEDSR"
wait
CUDA_VISIBLE_DEVICES=0,1,6,7 python -m torch.distributed.run --nproc_per_node=4 main.py --model_name "TCMF"
wait