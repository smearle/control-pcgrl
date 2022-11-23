from control_pcgrl.envs.probs.binary.binary_prob import BinaryProblem
from control_pcgrl.envs.probs.binary.binary_holey_prob import BinaryHoleyProblem
from control_pcgrl.envs.probs.ddave.ddave_prob import DDaveProblem
from control_pcgrl.envs.probs.mdungeon.mdungeon_prob import MDungeonProblem
from control_pcgrl.envs.probs.sokoban.sokoban_prob import SokobanProblem
from control_pcgrl.envs.probs.sokoban.sokoban_ctrl_prob import SokobanCtrlProblem
from control_pcgrl.envs.probs.zelda.zelda_prob import ZeldaProblem
from control_pcgrl.envs.probs.zelda.zelda_ctrl_prob import ZeldaCtrlProblem
from control_pcgrl.envs.probs.smb.smb_prob import SMBProblem
from control_pcgrl.envs.probs.zelda.minizelda_prob import MiniZeldaProblem
from control_pcgrl.envs.probs.zelda.zelda_play_prob import ZeldaPlayProblem
from control_pcgrl.envs.probs.smb.smb_ctrl_prob import SMBCtrlProblem
from control_pcgrl.envs.probs.loderunner_prob import LoderunnerProblem
from control_pcgrl.envs.probs.loderunner_ctrl_prob import LoderunnerCtrlProblem
from control_pcgrl.envs.probs.face_prob import FaceProblem
from control_pcgrl.envs.probs.microstructure.microstructure_prob import MicroStructureProblem

from control_pcgrl.envs.probs.minecraft.utils import patch_grpc_evocraft_imports
patch_grpc_evocraft_imports()

from control_pcgrl.envs.probs.minecraft.minecraft_2D_maze_prob import Minecraft2DmazeProblem
from control_pcgrl.envs.probs.minecraft.minecraft_3D_maze_prob import Minecraft3DmazeProblem
from control_pcgrl.envs.probs.minecraft.minecraft_3D_rain import Minecraft3Drain
from control_pcgrl.envs.probs.minecraft.minecraft_3D_holey_maze_prob import Minecraft3DholeymazeProblem
from control_pcgrl.envs.probs.minecraft.minecraft_3D_holey_dungeon_prob import Minecraft3DholeyDungeonProblem
from control_pcgrl.envs.probs.minecraft.minecraft_3D_Parkour_prob import Minecraft3DParkourProblem
from control_pcgrl.envs.probs.minecraft.minecraft_3D_Parkour_ctrl_prob import Minecraft3DParkourCtrlProblem

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
    "minecraft_3D_rain": Minecraft3Drain,
    "minecraft_3D_holey_maze": Minecraft3DholeymazeProblem,
    "minecraft_3D_dungeon_holey": Minecraft3DholeyDungeonProblem,
    "minecraft_3D_parkour": Minecraft3DParkourProblem,
    "minecraft_3D_parkour_ctrl": Minecraft3DParkourCtrlProblem,
}
