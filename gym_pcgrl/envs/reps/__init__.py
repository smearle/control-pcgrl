import imp
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.reps.narrow_cast_rep import NarrowCastRepresentation
from gym_pcgrl.envs.reps.narrow_multi_rep import NarrowMultiRepresentation
from gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation
from gym_pcgrl.envs.reps.turtle_cast_rep import TurtleCastRepresentation
from gym_pcgrl.envs.reps.ca_rep import CARepresentation
from gym_pcgrl.envs.reps.narrow_3D_rep import Narrow3DRepresentation
from gym_pcgrl.envs.reps.turtle_3D_rep import Turtle3DRepresentation
from gym_pcgrl.envs.reps.wide_3D_rep import Wide3DRepresentation
from gym_pcgrl.envs.reps.ca_3D_rep import CA3DRepresentation

# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "narrowcast": NarrowCastRepresentation,
    "narrowmulti": NarrowMultiRepresentation,
    "narrow3D": Narrow3DRepresentation,
    "wide": WideRepresentation,
    "wide3D": Wide3DRepresentation,
    "turtle": TurtleRepresentation,
    "turtle3D": Turtle3DRepresentation,
    "turtlecast": TurtleCastRepresentation,
    "cellular": CARepresentation,
    "cellular3D": CA3DRepresentation,
}
