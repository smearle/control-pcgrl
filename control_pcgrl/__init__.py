from pdb import set_trace as TT
from gym.envs.registration import register
from control_pcgrl.envs.probs import PROBLEMS
from control_pcgrl.envs.probs.holey_prob import HoleyProblem
from control_pcgrl.envs.probs.problem import Problem3D
from control_pcgrl.envs.reps import REPRESENTATIONS

# Register all the problems with every different representation for the OpenAI GYM
for prob in PROBLEMS.keys():
    prob_cls = PROBLEMS[prob]
    if issubclass(prob_cls, HoleyProblem):
        if issubclass(prob_cls, Problem3D):
            entry_point = "control_pcgrl.envs:PcgrlHoleyEnv3D"
        else:
            entry_point = "control_pcgrl.envs:PcgrlHoleyEnv"
    # elif "play" in prob:  # Deprecated
    #     entry_point= "control_pcgrl.envs:PlayPcgrlEnv"
    elif issubclass(prob_cls, Problem3D):
        entry_point= "control_pcgrl.envs:PcgrlEnv3D"
    else:
        entry_point= "control_pcgrl.envs:PcgrlCtrlEnv"
    # elif "_ctrl" in prob:
        # entry_point='gym_pcgrl.envs:PcgrlCtrlEnv'
    # else:
        # entry_point='gym_pcgrl.envs:PcgrlEnv'

    for rep in REPRESENTATIONS.keys():
        id = '{}-{}-v0'.format(prob, rep)
        register(
            id=id,
            entry_point=entry_point,
            kwargs={"prob": prob, "rep": rep},

            # Need this when using newer versions of gym. But we also need to update rendering to use the new 
            # version of gym.
#               order_enforce=False,  

        )

# print(registry.all())