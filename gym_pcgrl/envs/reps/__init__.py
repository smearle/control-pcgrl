from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.reps.narrow_cast_rep import NarrowCastRepresentation
from gym_pcgrl.envs.reps.narrow_multi_rep import NarrowMultiRepresentation
from gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation
from gym_pcgrl.envs.reps.turtle_cast_rep import TurtleCastRepresentation
from gym_pcgrl.envs.reps.narrow_3D_rep import Narrow3DRepresentation

# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "narrowcast": NarrowCastRepresentation,
    "narrowmulti": NarrowMultiRepresentation,
    "narrow3D": Narrow3DRepresentation,
    "wide": WideRepresentation,
    "turtle": TurtleRepresentation,
    # "turtle3D": Turtle3D,
    # "wide3D": Wide3D,
    "turtlecast": TurtleCastRepresentation
}
