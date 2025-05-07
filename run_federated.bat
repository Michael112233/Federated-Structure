python src/federated_main.py --model=mlp --dataset=cifar --epochs=30 --algorithm=FedProx-CC --iid=0 --lr=0.1 --cluster=4
python src/federated_main.py --model=mlp --dataset=cifar --epochs=30 --algorithm=FedProx-CC --iid=1 --lr=0.1 --cluster=2
python src/federated_main.py --model=mlp --dataset=mnist --epochs=30 --algorithm=FedProx-CC --iid=0 --lr=0.1 --cluster=4
python src/federated_main.py --model=mlp --dataset=mnist --epochs=30 --algorithm=FedProx-CC --iid=1 --lr=0.1 --cluster=2
python src/federated_main.py --model=mlp --dataset=fmnist --epochs=30 --algorithm=FedProx-CC --iid=0 --lr=0.1 --cluster=4
python src/federated_main.py --model=mlp --dataset=fmnist --epochs=30 --algorithm=FedProx-CC --iid=1 --lr=0.1 --cluster=2
