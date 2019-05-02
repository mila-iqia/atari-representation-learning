import subprocess
import sys
base_cmd = "sbatch"
ss= "exp_scripts/run_gpu_cloud.sl"
module = "scripts.run_probe"
args = [base_cmd, ss, module]
args.extend(sys.argv[1:])
zoo = True
if zoo:
    args.append("--probe-collect-mode atari_zoo")
    args.append('--zoo-algos apex es a2c dqn rainbow')
    args.append('--zoo-tags final 1B 400M')
else:
    pass

args.append("--train-encoder")
args.append('--no-downsample')
args.append('--num-frame-stack 1')
args.append('--num-runs 3')

envs1 =  ['asteroids', 'berzerk', 'boxing',
        'demon_attack', 'enduro', 'freeway', 'frostbite']
        
envs2 = ['hero', 'ms_pacman', 'pong', 'private_eye','qbert', 'riverraid', 'seaquest']

envs3 = ['solaris', 'space_invaders', 'venture', 'video_pinball', 'yars_revenge','breakout','pitfall','montezuma_revenge']

all_envs = envs1 + envs2 + envs3

envs = all_envs
suffix = "NoFrameskip-v4"
for i,env in enumerate(envs):
    
    names = env.split("_")
    name = "".join([s.capitalize() for s in names])
    sargs = args + ["--env-name"]
    
    sargs.append(name + suffix) 
    
    print(sargs) 
    subprocess.run(sargs)
