import os

def eval(dir):
    
    for filename in os.listdir(dir):
        argslist = []
        if filename.endswith("log"):
            argslist.append(filename.split("_")[:-1])

        for arg in argslist:
            controls = arg[4:-1]
            command = (f"python evaluate_ctrl.py -p {arg[0]}_{arg[1]} -r {arg[2]} -c ")
            for condition in controls:
                if condition == 'ALPGMM':
                    command += '--alp_gmm '
                else:
                    command +=  condition + " "
            command += "--n_maps 3 --n_trials 3 --n_bins 3 --diversity_eval --render_level "

            print(command)
            os.system(command)
                 
if __name__ == '__main__':
    eval('../runs')
    