from gym_pcgrl.envs.probs.binary.binary_prob import BinaryProblem
from gym_pcgrl.envs.probs.ddave.ddave_prob import DDaveProblem
from gym_pcgrl.envs.probs.mdungeon.mdungeon_prob import MDungeonProblem
from gym_pcgrl.envs.probs.sokoban.sokoban_prob import SokobanProblem
from gym_pcgrl.envs.probs.zelda.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.probs.smb.smb_prob import SMBProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_2Dmaze_prob import Minecraft2DmazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3Dmaze_prob import Minecraft3DmazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3DZelda_prob import Minecraft3DZeldaProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "binary": BinaryProblem,
    "ddave": DDaveProblem,
    "mdungeon": MDungeonProblem,
    "sokoban": SokobanProblem,
    "zelda": ZeldaProblem,
    "smb": SMBProblem,
    "minecraft_2D_maze": Minecraft2DmazeProblem,
    "minecraft_3D_maze": Minecraft3DmazeProblem,
    "minecraft_3D_zelda": Minecraft3DZeldaProblem
}
