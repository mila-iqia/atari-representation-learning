import subprocess

base_cmd = "sbatch"
ss= "exp_scripts/run_gpu_cloud.sl"
module = "scripts.run_probe"
args = [base_cmd, ss, module]
args.append("--method pretrained-rl-agent")
#args.append("--probe-collect-mode random_agent")
args.append("--probe-collect-mode pretrained_ppo")
checkpoints = [33]
envs = ['asteroids',
 'berzerk',
 'bowling',
 'boxing',
 'breakout',
 'demonattack',
 'freeway',
 'frostbite',
 'hero',
 'montezumarevenge',
 'mspacman',
 'pitfall',
 'pong',
 'privateeye',
 'qbert',
 'riverraid',
 'seaquest',
 'spaceinvaders',
 'tennis',
 'venture',
 'videopinball',
 'yarsrevenge']


suffix = "NoFrameskip-v4"
for i,env in enumerate(envs):
    for ind in checkpoints:
        names = env.split("_")
        name = "".join([s.capitalize() for s in names])
        sargs = args + ["--env-name"]
    
        sargs.append(name + suffix) 
        sargs.extend(["--checkpoint-index",str(ind)])
    
        print(" ".join(sargs))
        subprocess.run(sargs)
