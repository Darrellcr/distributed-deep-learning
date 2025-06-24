from torch.distributed.pipelining import SplitPoint, pipeline
import torch
from torchvision import models


model = models.densenet121()
print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

# Backward of skip connections not supported yet, e.g. pp-nnpkwxs5t9sgh
# pipe = pipeline(
#     module=model,
#     mb_args=(torch.rand(1, 3, 224, 224),),
#     split_spec={
#         "features.denseblock3.denselayer1": SplitPoint.BEGINNING,
#         "features.denseblock3.denselayer22": SplitPoint.BEGINNING,
#         "features.denseblock4.denselayer8": SplitPoint.BEGINNING,
#     }
# )

# job id pp-ssbwpk61xtdchc (training time got slower)
pipe = pipeline(
    module=model,
    mb_args=(torch.rand(1, 3, 224, 224),),
    split_spec={
        "features.denseblock2.denselayer12": SplitPoint.END,
        "features.denseblock3.denselayer24": SplitPoint.END,
        "features.denseblock4.denselayer16": SplitPoint.END,
    }
)

# job id pp-ghp6hmmmz9x2zc
pipe = pipeline(
    module=model,
    mb_args=(torch.rand(1, 3, 224, 224),),
    split_spec={
        "features.denseblock2.denselayer1": SplitPoint.BEGINNING,
        "features.denseblock3.denselayer1": SplitPoint.BEGINNING,
        "features.denseblock4.denselayer1": SplitPoint.BEGINNING,
    }
)

# print(pipe.get_stage_module(1))
# num parameters

total = 0
for i in range(pipe.num_stages):
    stage_num_params = sum(p.numel()
                           for p in pipe.get_stage_module(i).parameters())
    total += stage_num_params
    print(f"Stage {i} num params: {stage_num_params}")

print(f"Total num params: {total}")
# print(pipe.get_stage_module(0))
# print(pipe.get_stage_module(0).graph)
