import os

def eval(dir):

    argslist = []

    for filename in os.listdir(dir):
        if filename.endswith("log"):
            argslist.append(filename.split("_")[:-1])


    for arg in argslist:
        conditionals = arg[3:-1]
        command = (f"python evaluate.py -p {arg[0]} -r {arg[1]} -c ")
        for condition in conditionals:
            command +=  condition + " "
        command += f"--load_best --experiment_id {arg[-1]} --n_maps 6"
        os.system(command)
                 

if __name__ == '__main__':
    eval('runs')

