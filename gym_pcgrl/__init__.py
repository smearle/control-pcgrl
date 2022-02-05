from gym.envs.registration import register
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS

# Register all the problems with every different representation for the OpenAI GYM
for prob in PROBLEMS.keys():
    for rep in REPRESENTATIONS.keys():
        if ("3D" not in prob) and ("3D" not in rep):
            register(
                id='{}-{}-v0'.format(prob, rep),
                entry_point='gym_pcgrl.envs:PcgrlEnv',
                kwargs={"prob": prob, "rep": rep}
            )
        elif ("3D" in prob) and ("3D" in rep):
            register(
                id='{}-{}-v0'.format(prob, rep),
                entry_point="gym_pcgrl.envs:PcgrlEnv3D",
                kwargs={"prob": prob, "rep": rep}
            )
        else:
            continue
