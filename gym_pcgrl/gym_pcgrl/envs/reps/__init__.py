from gym_pcgrl.envs.reps.ca_3D_holey import CA3DRepresentationHoley
from gym_pcgrl.envs.reps.narrow_rep import NarrowRepresentation
from gym_pcgrl.envs.reps.narrow_cast_rep import NarrowCastRepresentation
from gym_pcgrl.envs.reps.narrow_multi_rep import NarrowMultiRepresentation
from gym_pcgrl.envs.reps.narrow_holey_rep import NarrowHoleyRepresentation
from gym_pcgrl.envs.reps.narrow_3D_rep import Narrow3DRepresentation
from gym_pcgrl.envs.reps.narrow_3D_holey_rep import Narrow3DHoleyRepresentation
from gym_pcgrl.envs.reps.turtle_rep import TurtleRepresentation
from gym_pcgrl.envs.reps.turtle_holey_rep import TurtleHoleyRepresentation
from gym_pcgrl.envs.reps.turtle_cast_rep import TurtleCastRepresentation
from gym_pcgrl.envs.reps.turtle_3D_rep import Turtle3DRepresentation
from gym_pcgrl.envs.reps.turtle_3D_holey_rep import Turtle3DHoleyRepresentation
from gym_pcgrl.envs.reps.wide_rep import WideRepresentation
from gym_pcgrl.envs.reps.wide_holey_rep import WideHoleyRepresentation
from gym_pcgrl.envs.reps.wide_3D_rep import Wide3DRepresentation
from gym_pcgrl.envs.reps.wide_3D_holey_rep import Wide3DHoleyRepresentation
from gym_pcgrl.envs.reps.ca_rep import CARepresentation
from gym_pcgrl.envs.reps.ca_3D_rep import CA3DRepresentation


# all the representations should be defined here with its corresponding class
REPRESENTATIONS = {
    "narrow": NarrowRepresentation,
    "narrowcast": NarrowCastRepresentation,
    "narrowmulti": NarrowMultiRepresentation,
    "narrowholey": NarrowHoleyRepresentation,
    "narrow3D": Narrow3DRepresentation,
    "narrow3Dholey": Narrow3DHoleyRepresentation,
    "turtle": TurtleRepresentation,
    "turtlecast": TurtleCastRepresentation,
    "turtleholey": TurtleHoleyRepresentation,
    "turtle3D": Turtle3DRepresentation,
    "turtle3Dholey": Turtle3DHoleyRepresentation,
    "wide": WideRepresentation,
    "wideholey": WideHoleyRepresentation,
    "wide3D": Wide3DRepresentation,
    "wide3Dholey": Wide3DHoleyRepresentation,
    "cellular": CARepresentation,
    "cellular3D": CA3DRepresentation,
    "cellular3Dholey": CA3DRepresentationHoley,
}
