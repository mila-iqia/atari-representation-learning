import subprocess
import sys
base_cmd = "sbatch"
ss= "exp_scripts/run_gpu_azure.sl"
module = "scripts.run_probe"
args = [base_cmd, ss, module]
args.append("--method {}".format(sys.argv[1]))
args.append("--probe-collect-mode atari_zoo")
args.append('--zoo-algos apex es a2c dqn rainbow')
args.append('--zoo-tags final 1B 400M')
args.append('--no-downsample')
args.append('--num-frame-stack 1')
args.append('--num-runs 5')

envs =  ['asteroids', 'berzerk', 'boxing',
        'demon_attack', 'enduro', 'freeway', 'frostbite', 'hero', 
        'ms_pacman', 'pong', 'private_eye', 'qbert', 'riverraid', 
        'seaquest', 'solaris', 'space_invaders', 'venture', 'video_pinball', 
        'yars_revenge','breakout','pitfall','montezuma_revenge'
        ]


suffix = "NoFrameskip-v4"
for i,env in enumerate(envs):
    
    names = env.split("_")
    name = "".join([s.capitalize() for s in names])
    sargs = args + ["--env-name"]
    
    sargs.append(name + suffix) 
    
    print(sargs) 
    subprocess.run(sargs)
