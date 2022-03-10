from gym_pcgrl.envs.probs.binary.binary_prob import BinaryProblem
from gym_pcgrl.envs.probs.binary.binary_ctrl_prob import BinaryCtrlProblem
from gym_pcgrl.envs.probs.ddave.ddave_prob import DDaveProblem
from gym_pcgrl.envs.probs.mdungeon.mdungeon_prob import MDungeonProblem
from gym_pcgrl.envs.probs.sokoban.sokoban_prob import SokobanProblem
from gym_pcgrl.envs.probs.sokoban.sokoban_ctrl_prob import SokobanCtrlProblem
from gym_pcgrl.envs.probs.zelda.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.probs.zelda.zelda_ctrl_prob import ZeldaCtrlProblem
from gym_pcgrl.envs.probs.smb.smb_prob import SMBProblem
from gym_pcgrl.envs.probs.zelda.minizelda_prob import MiniZeldaProblem
from gym_pcgrl.envs.probs.zelda.zelda_play_prob import ZeldaPlayProblem
from gym_pcgrl.envs.probs.smb.smb_ctrl_prob import SMBCtrlProblem
from gym_pcgrl.envs.probs.loderunner_prob import LoderunnerProblem
from gym_pcgrl.envs.probs.loderunner_ctrl_prob import LoderunnerCtrlProblem
from gym_pcgrl.envs.probs.face_prob import FaceProblem
from gym_pcgrl.envs.probs.microstructure_prob import MicroStructureProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_2Dmaze_prob import Minecraft2DmazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3Dmaze_prob import Minecraft3DmazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3Dmaze_ctrl_prob import Minecraft3DmazeCtrlProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3DZelda_prob import Minecraft3DZeldaProblem

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
    "minecraft_3D_maze_ctrl": Minecraft3DmazeCtrlProblem,
    "minecraft_3D_zelda": Minecraft3DZeldaProblem
}
