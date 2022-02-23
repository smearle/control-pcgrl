from gym_pcgrl.envs.probs.binary_prob import BinaryProblem
from gym_pcgrl.envs.probs.binary_ctrl_prob import BinaryCtrlProblem
from gym_pcgrl.envs.probs.ddave_prob import DDaveProblem
from gym_pcgrl.envs.probs.mdungeon_prob import MDungeonProblem
from gym_pcgrl.envs.probs.sokoban_prob import SokobanProblem
from gym_pcgrl.envs.probs.sokoban_ctrl_prob import SokobanCtrlProblem
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.probs.zelda_ctrl_prob import ZeldaCtrlProblem
from gym_pcgrl.envs.probs.smb_prob import SMBProblem
from gym_pcgrl.envs.probs.minizelda_prob import MiniZeldaProblem
from gym_pcgrl.envs.probs.zelda_play_prob import ZeldaPlayProblem
from gym_pcgrl.envs.probs.smb_ctrl_prob import SMBCtrlProblem
from gym_pcgrl.envs.probs.loderunner_prob import LoderunnerProblem
from gym_pcgrl.envs.probs.loderunner_ctrl_prob import LoderunnerCtrlProblem
from gym_pcgrl.envs.probs.face_prob import FaceProblem
from gym_pcgrl.envs.probs.microstructure_prob import MicroStructureProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_2Dmaze_prob import Minecraft2DmazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3Dmaze_prob import Minecraft3DmazeProblem
# from gym_pcgrl.envs.probs.minecraft.minecraft_3Dmaze_ctrl_prop import Minecraft3DmazeCtrlProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "binary": BinaryProblem,
    "binary_ctrl": BinaryCtrlProblem,
    "ddave": DDaveProblem,
    "mdungeon": MDungeonProblem,
    "sokoban": SokobanProblem,
    "sokoban_ctrl": SokobanCtrlProblem,
    "zelda": ZeldaProblem,
    "smb": SMBProblem,
    "mini": MiniZeldaProblem,
    "zeldaplay": ZeldaPlayProblem,
    "zelda_ctrl": ZeldaCtrlProblem,
    "smb_ctrl": SMBCtrlProblem,
    "loderunner": LoderunnerProblem,
    "loderunner_ctrl": LoderunnerCtrlProblem,
    "face_ctrl": FaceProblem,
    "microstructure": MicroStructureProblem,
    "minecraft_2D_maze": Minecraft2DmazeProblem,
    "minecraft_3D_maze": Minecraft3DmazeProblem,
    # "minecraft_3D_maze_ctrl": Minecraft3DmazeCtrlProblem
}
