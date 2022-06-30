from gym_pcgrl.envs.probs.binary.binary_prob import BinaryProblem
from gym_pcgrl.envs.probs.binary.binary_holey_prob import BinaryHoleyProblem
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
from gym_pcgrl.envs.probs.microstructure.microstructure_prob import MicroStructureProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_2D_maze_prob import Minecraft2DmazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3D_maze_prob import Minecraft3DmazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3D_holey_maze_prob import Minecraft3DholeymazeProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3D_holey_dungeon_prob import Minecraft3DholeyDungeonProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3D_Parkour_prob import Minecraft3DParkourProblem
from gym_pcgrl.envs.probs.minecraft.minecraft_3D_Parkour_ctrl_prob import Minecraft3DParkourCtrlProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "binary": BinaryProblem,
    "binary_holey": BinaryHoleyProblem,
    "ddave": DDaveProblem,
    "mdungeon": MDungeonProblem,
    "sokoban": SokobanCtrlProblem,
    # "sokoban_ctrl": SokobanCtrlProblem,
    # "zelda": ZeldaProblem,
    "smb": SMBCtrlProblem,
    "mini": MiniZeldaProblem,
    "zeldaplay": ZeldaPlayProblem,
    "zelda": ZeldaCtrlProblem,
    "smb_ctrl": SMBCtrlProblem,
    "loderunner": LoderunnerCtrlProblem,
    "loderunner_ctrl": LoderunnerCtrlProblem,
    "face_ctrl": FaceProblem,
    "microstructure": MicroStructureProblem,
    "minecraft_2D_maze": Minecraft2DmazeProblem,
    "minecraft_3D_maze": Minecraft3DmazeProblem,
    "minecraft_3D_holey_maze": Minecraft3DholeymazeProblem,
    "minecraft_3D_dungeon_holey": Minecraft3DholeyDungeonProblem,
    "minecraft_3D_parkour": Minecraft3DParkourProblem,
    "minecraft_3D_parkour_ctrl": Minecraft3DParkourCtrlProblem,
}
