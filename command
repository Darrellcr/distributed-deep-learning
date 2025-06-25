### single with ddp
torchx run -s kubernetes dist.ddp -j 1x1 --script /mnt/dcornelius/code/distributed-deep-learning/ddp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --gpu 1 --memMB 16384 --image darrellcr/ddpnpp:v2

### single
torchx run -s kubernetes ./component/single.py:python --script /mnt/dcornelius/code/distributed-deep-learning/single.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 1 --memMB 16384 --image darrellcr/ddpnpp:v2

### ddp
torchx run -s kubernetes dist.ddp -j 1x2 --script /mnt/dcornelius/code/distributed-deep-learning/ddp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 2 --memMB 16384 --image darrellcr/ddpnpp:v2

torchx run -s kubernetes dist.ddp -j 2x1 --script /mnt/dcornelius/code/distributed-deep-learning/ddp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 1 --memMB 16384 --image darrellcr/ddpnpp:v2
torchx run -s kubernetes dist.ddp -j 3x2 --script /mnt/dcornelius/code/distributed-deep-learning/ddp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 2 --memMB 16384 --image darrellcr/ddpnpp:v2
torchx run -s kubernetes dist.ddp -j 3x1 --script /mnt/dcornelius/code/distributed-deep-learning/ddp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 1 --memMB 16384 --image darrellcr/ddpnpp:v2


### pp 
torchx run -s kubernetes dist.ddp -j 1x2 --script /mnt/dcornelius/code/distributed-deep-learning/pp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 2 --memMB 16384 --image darrellcr/ddpnpp:v2

torchx run -s kubernetes dist.ddp -j 2x1 --script /mnt/dcornelius/code/distributed-deep-learning/pp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 1 --memMB 16384 --image darrellcr/ddpnpp:v2
torchx run -s kubernetes dist.ddp -j 2x2 --script /mnt/dcornelius/code/distributed-deep-learning/pp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 2 --memMB 16384 --image darrellcr/ddpnpp:v2

### ddp n pp
torchx run -s kubernetes dist.ddp -j 3x2 --script /mnt/dcornelius/code/distributed-deep-learning/ddp_n_pp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 2 --memMB 16384 --image darrellcr/ddpnpp:v2

torchx run -s kubernetes dist.ddp -j 2x2 --script /mnt/dcornelius/code/distributed-deep-learning/ddp_n_pp.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 2 --memMB 16384 --image darrellcr/ddpnpp:v2

### running test.py
torchx run -s kubernetes dist.ddp -j 3x2 --script /mnt/dcornelius/code/distributed-deep-learning/test.py --mount type=volume,src=nas-pvc,dst=/mnt/dcornelius --cpu 4 --gpu 2 --memMB 8192 --image darrellcr/ddpnpp:v2
