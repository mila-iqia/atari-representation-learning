import wget
from pathlib import Path
import numpy as np
from src.atari import convert_ram_to_label
import cv2
def separate_into_episodes(arr, ep_inds):
    middle_eps = [arr[ep_inds[i]:ep_inds[i+1]] for i in range(0,len(ep_inds)-1)] # gets all episodes except the first and last
    first_ep, last_ep = arr[:ep_inds[0]], arr[ep_inds[-1]:]
    episodes =  [first_ep] + middle_eps + [last_ep]
    return episodes

def convert2grayscale(frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = np.expand_dims(frame, -1)
        return frame
def get_atari_zoo_episodes(env,tags=["pretraining-only"], num_frame_stack=4, downsample=True):
    #using atari zoo (https://github.com/uber-research/atari-model-zoo) url. Thanks, Uber!
    base_url = "https://dgqeqexrlnkvd.cloudfront.net/zoo"
    if "pretraining-only" in tags:
        run_ids = [1,2]
    elif "probe-only" in tags:
        run_ids = [3]
    basepath = Path("./data")

    basepath.mkdir(parents=True,exist_ok=True)
    algos = ["a2c","apex","ga","es"]
    tags = ["initial","1HR","2HR", "6HR", "10HR","final", "400M", "1B"]

    representations, ep_rewards, scores, observations, rams, frames, labels = [],[],[],[],[],[],[]
    
    for algo in algos:
        for tag in tags:
            if (algo == "dqn" or algo == "rainbow") and tag != "final":
                assert False, "DQN and Rainbow only work with the tag \"final\" not the one you put: \"{}\"".format(tag)
            for run_id in run_ids:
                fname = "_".join([env,algo,str(run_id),tag]) + ".npz"
                savepath = basepath/fname

                if not savepath.exists():
                    final_url = "/".join([base_url,algo,env,"model{}_{}_rollout.npz".format(run_id, tag)])
                    wget.download(final_url, str(savepath))

                fnp = np.load(savepath)

                cur_reps, cur_ep_rewards, cur_scores, cur_obs, cur_rams, cur_frames = [fnp[k] for k in ['representation',
                                                                            'ep_rewards', 
                                                                            'score', 
                                                                            'observations', 
                                                                            'ram',
                                                                            'frames']]
                
     
                cur_labels = [convert_ram_to_label(env,ram) for ram in cur_rams]
                #get index where the ep
                # aka indices in which the last 3 frames of the observation at i-1 do not equal the first 3 frames of the observation at i
                diff = cur_obs[1:,:,:,:3] - cur_obs[:-1,:,:,1:]
                diff_arr = np.asarray([int(np.all(d==0)) for d in diff])
                inds = np.arange(len(diff_arr))
                ep_inds = inds[diff_arr < 1] + 1 #add one to offset the off by one error
                
                cur_frames = [convert2grayscale(frame) for frame in cur_frames]
                
                rep_eps = separate_into_episodes(cur_reps, ep_inds)
                score_eps = separate_into_episodes(cur_scores, ep_inds)
                obs_eps = separate_into_episodes(cur_obs, ep_inds)
                ram_eps = separate_into_episodes(cur_rams, ep_inds)
                frame_eps = separate_into_episodes(cur_frames, ep_inds)
                label_eps = separate_into_episodes(cur_labels, ep_inds)
                
                representations.extend(rep_eps)
                ep_rewards.extend(cur_ep_rewards)
                scores.extend(score_eps)
                observations.extend(obs_eps)
                rams.extend(ram_eps)
                frames.extend(frame_eps)
                labels.extend(label_eps)
                
                if downsample==True:
                    if num_frame_stack == 4:
                        episodes = observations
                    elif num_frame_stack == 1:
                        assert False, "No you can't do num frame stack {} and downsample {}".format(num_frame_stack, downsample)
                else:
                    if num_frame_stack == 1:
                        episodes = frames
                    else:
                        assert False, "No you can't do num frame stack {} and downsample {}".format(num_frame_stack, downsample)
                        
                        
                
                
    return episodes, labels