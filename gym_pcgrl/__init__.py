from gym.envs.registration import register
from gym_pcgrl.envs.probs import PROBLEMS
from gym_pcgrl.envs.reps import REPRESENTATIONS

# Register all the problems with every different representation for the OpenAI GYM
for prob in PROBLEMS.keys():
    if "_ctrl" in prob:
        for rep in REPRESENTATIONS.keys():
            register(
                id='{}-{}-v0'.format(prob, rep),
                entry_point='gym_pcgrl.envs:PcgrlCtrlEnv',
                kwargs={"prob": prob, "rep": rep}
            )
    else:
        for rep in REPRESENTATIONS.keys():
            register(
                id='{}-{}-v0'.format(prob, rep),
                entry_point='gym_pcgrl.envs:PcgrlEnv',
                kwargs={"prob": prob, "rep": rep}
            )
