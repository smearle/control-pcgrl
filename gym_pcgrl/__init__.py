from gym.envs.registration import register
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS

# Register all the problems with every different representation for the OpenAI GYM
for prob in PROBLEMS.keys():
    if prob == "minecraft_3D_maze":
        continue
    for rep in REPRESENTATIONS.keys():
        if rep == "narrow3D":
            continue
        register(
            id='{}-{}-v0'.format(prob, rep),
            entry_point='gym_pcgrl.envs:PcgrlEnv',
            kwargs={"prob": prob, "rep": rep}
        )

register(
    id="minecraft_3D_maze-narrow3D-v0",
    entry_point="gym_pcgrl.envs:PcgrlEnv3D",
    kwargs={"prob": "minecraft_3D_maze", "rep": "narrow3D"}
)
