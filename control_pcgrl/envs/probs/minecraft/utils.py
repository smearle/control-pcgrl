from pdb import set_trace as TT
import os


def patch_grpc_evocraft_imports():
    # HACK: Replace relative import
    # Get parent of parent of this file
    gym_pcgrl_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    fp = os.path.join(gym_pcgrl_dir, "minecraft/minecraft_pb2_grpc.py")
    with open(fp, "r") as f:
        contents = f.read()
    contents = contents.replace("from src.main.proto", "from control_pcgrl.envs.probs.minecraft")
    # Write file contents
    with open(fp, "w") as f:
        f.write(contents)