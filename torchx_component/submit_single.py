from torchx import specs


# def submit(script: str, cpu: int, memMB: int) -> specs.AppDef:
#     return specs.AppDef(
#         name="single",
#         roles=[
#             specs.Role(
#                 name="train",
#                 image="darrellcr/ddpnpp:v2",
#                 resource=specs.Resource(
#                     cpu=cpu,
#                     memMB=memMB,
#                     gpu=1,
#                     capabilities={
#                         "kubernetes.io/hostname": "cn2"
#                     },
#                 ),
#                 mounts=[
#                     specs.VolumeMount(
#                         src="nas-pvc",
#                         dst_path="/mnt/dcornelius",
#                     ),
#                 ],
#                 entrypoint="python",
#                 args=[
#                     script,
#                 ]
#             )
#         ],
#     )

import torchx
from torchx import specs
from typing import Optional, List

def python(                                                                                                                
    *args: str,                                                                                                            
    m: Optional[str] = None,                                                                                               
    c: Optional[str] = None,                                                                                               
    script: Optional[str] = None,                                                                                          
    image: str = torchx.IMAGE,                                                                                             
    name: str = "torchx_utils_python",                                                                                     
    cpu: int = 1,                                                                                                          
    gpu: int = 0,                                                                                                          
    memMB: int = 1024,                                                                                                     
    h: Optional[str] = None,                                                               
    num_replicas: int = 1,                                                                 
    mounts: Optional[List[str]] = None                                                         
) -> specs.AppDef:                                                                                                                                   
    if sum([m is not None, c is not None, script is not None]) != 1:                           
        raise ValueError(                                                        
            "exactly one of `-m`, `-c` and `--script` needs to be specified"                                               
        )                                                                                      
                                                                                           
    if script:                                                                                                             
        cmd = [script]                                                          
    elif m:                                                                                                                
        cmd = ["-m", m]                                                                        
    elif c:                                                                                                                
        cmd = ["-c", c]                                                                                                    
    else:                                                                                  
        raise ValueError("no program specified")                                       
                                                                                               
    return specs.AppDef(                                                                   
        name=name,                                                                                                         
        roles=[                                                        
            specs.Role(                                                                        
                name="python",                                                                 
                image=image,                                                                                               
                entrypoint="python",                                                                                       
                num_replicas=num_replicas,                                                     
                resource=specs.resource(cpu=cpu, gpu=gpu, memMB=memMB, h=h),               
                args=[*cmd, *args],                                                                                        
                env={"HYDRA_MAIN_MODULE": m} if m else {},                             
                mounts=specs.parse_mounts(mounts) if mounts else [],                   
            )                                                                     
        ],                                                                                 
    ) 