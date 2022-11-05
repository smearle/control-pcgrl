from pdb import set_trace as TT
import os


def patch_grpc_evocraft_imports():
    # HACK: Replace relative import
    with open("gym_pcgrl/gym_pcgrl/envs/probs/minecraft/minecraft_pb2_grpc.py", "r") as f:
        contents = f.read()
    contents = contents.replace("from src.main.proto", "from gym_pcgrl.envs.probs.minecraft")
    # Write file contents
    with open("gym_pcgrl/gym_pcgrl/envs/probs/minecraft/minecraft_pb2_grpc.py", "w") as f:
        f.write(contents)