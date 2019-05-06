import wget
from pathlib import Path
import numpy as np
from src.atari import convert_ram_to_label
import cv2
import sys
def separate_into_episodes(arr, ep_inds):
    middle_eps = [arr[ep_inds[i]:ep_inds[i+1]] for i in range(0,len(ep_inds)-1)] # gets all episodes except the first and last
    first_ep, last_ep = arr[:ep_inds[0]], arr[ep_inds[-1]:]
    episodes =  [first_ep] + middle_eps + [last_ep]
    return episodes

def get_episode_inds(obs):
    #get index where the ep
    # aka indices in which the last 3 frames of the observation at i-1 do not equal the first 3 frames of the observation at i
    diff = obs[1:,:,:,:3] - obs[:-1,:,:,1:]
    diff_arr = np.asarray([int(np.all(d==0)) for d in diff])
    inds = np.arange(len(diff_arr))
    ep_inds = inds[diff_arr < 1] + 1 #add one to offset the off by one error
    return ep_inds
    
def convert2grayscale(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        return frame
def get_atari_zoo_episodes(env, run_ids="all", num_frame_stack=4, downsample=True, algos = ["a2c","apex","ga","es"],
                           tags = ["initial","1HR","2HR", "6HR", "10HR","final", "400M", "1B"],
                           use_representations_instead_of_frames=False):
    
    #using atari zoo (https://github.com/uber-research/atari-model-zoo) url. Thanks, Uber!
    base_url = "https://dgqeqexrlnkvd.cloudfront.net/zoo"

    basepath = Path("./data")

    basepath.mkdir(parents=True,exist_ok=True)
   

    episodes, episode_labels, episode_rewards = [],[],[]
    
    for algo in algos:
        for tag in tags:
            if (algo == "dqn" or algo == "rainbow") and tag != "final":
                sys.stderr.write("DQN and Rainbow only work with the tag \"final\" not the one you put: \"{}\"".format(tag))
                continue

            if run_ids == "all":
                if algo in ["apex", "dqn", "rainbow"]:
                    run_ids = [1,2,3,4,5]
                else:
                    run_ids = [1,2,3]
                
            for run_id in run_ids:
                fname = "_".join([env,algo,str(run_id),tag]) + ".npz"
                savepath = basepath/fname

                if not savepath.exists():
                    final_url = "/".join([base_url,algo,env,"model{}_{}_rollout.npz".format(run_id, tag)])
                    try:
                        wget.download(final_url, str(savepath))
                    except:
                        sys.stderr.write("Unable to download {}. Skipping... On to Cincinnati...".format(final_url))
                        continue
                try:
                    fnp = np.load(savepath)
                    cur_obs, cur_rams, cur_frames, ep_rewards = fnp['observations'], fnp['ram'], fnp['frames'], fnp['ep_rewards']
                except:
                    sys.stderr.write("Had trouble opening {}. Skipping this one for now...".format(savepath))
                    continue
                    
                
          
                ep_inds = get_episode_inds(cur_obs)
                
                cur_labels = [convert_ram_to_label(env,ram) for ram in cur_rams]
                label_eps = separate_into_episodes(cur_labels, ep_inds)
                
                episode_labels.extend(label_eps)
                episode_rewards.extend(ep_rewards)
                
                
                if use_representations_instead_of_frames:
                    try:
                        cur_reps = fnp["representation"]
                    except:
                        sys.stderr.write("Had trouble opening {}[\"representation\"]. Skipping this one for now...".format(savepath))
                        continue
                    rep_eps = separate_into_episodes(cur_reps, ep_inds)
                    episodes.extend(rep_eps)
                
                
                
                else:
                    if downsample==True and num_frame_stack == 4:
                            eps = separate_into_episodes(cur_obs, ep_inds)

                    elif downsample==False and  num_frame_stack == 1:
                            cur_frames = [convert2grayscale(frame) for frame in cur_frames]
                            eps = separate_into_episodes(cur_frames, ep_inds)
                            eps = [np.asarray(ep) for ep in eps]
                    else:
                        assert False, "No you can't do num frame stack {} and downsample {}".format(num_frame_stack, downsample)

                    episodes.extend(eps)
                
         
    return episodes, episode_labels, episode_rewards
