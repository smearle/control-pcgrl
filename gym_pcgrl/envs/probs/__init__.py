from gym_pcgrl.envs.probs.binary_prob import BinaryProblem
from gym_pcgrl.envs.probs.binary_ctrl_prob import BinaryCtrlProblem
from gym_pcgrl.envs.probs.ddave_prob import DDaveProblem
from gym_pcgrl.envs.probs.mdungeon_prob import MDungeonProblem
from gym_pcgrl.envs.probs.sokoban_prob import SokobanProblem
from gym_pcgrl.envs.probs.sokoban_ctrl_prob import SokobanCtrlProblem
from gym_pcgrl.envs.probs.zelda_prob import ZeldaProblem
from gym_pcgrl.envs.probs.zelda_ctrl_prob import ZeldaCtrlProblem
from gym_pcgrl.envs.probs.smb_prob import SMBProblem

# all the problems should be defined here with its corresponding class
PROBLEMS = {
    "binary": BinaryProblem,
    "binary_ctrl": BinaryCtrlProblem,
    "ddave": DDaveProblem,
    "mdungeon": MDungeonProblem,
    "sokoban": SokobanProblem,
    "sokoban_ctrl": SokobanCtrlProblem,
    "zelda": ZeldaProblem,
    "zelda_ctrl": ZeldaCtrlProblem,
    "smb": SMBProblem,
}
