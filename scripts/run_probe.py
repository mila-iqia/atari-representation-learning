from scripts.run_contrastive import train_encoder
from atariari.benchmark.probe import train_all_probes, postprocess_raw_metrics

import torch
from atariari.methods.utils import get_argparser, train_encoder_methods, probe_only_methods
from atariari.methods.encoders import NatureCNN, ImpalaCNN
import wandb
import sys
from atariari.methods.majority import majority_baseline
from atariari.benchmark.episodes import get_episodes





def run_probe(args):
    wandb.config.update(vars(args))
    tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels = get_episodes(steps=args.probe_steps,
                                                                                 env_name=args.env_name,
                                                                                 seed=args.seed,
                                                                                 num_processes=args.num_processes,
                                                                                 num_frame_stack=args.num_frame_stack,
                                                                                 downsample=not args.no_downsample,
                                                                                 color=args.color,
                                                                                 entropy_threshold=args.entropy_threshold,
                                                                                 collect_mode=args.probe_collect_mode,
                                                                                 train_mode="probe",
                                                                                 checkpoint_index=args.checkpoint_index,
                                                                                 min_episode_length=args.batch_size)

    print("got episodes!")





    if args.train_encoder and args.method in train_encoder_methods:
        print("Training encoder from scratch")
        encoder = train_encoder(args)
        encoder.probing = True
        encoder.eval()

    elif args.method in ["pretrained-rl-agent", "majority"]:
        encoder = None

    else:
        observation_shape = tr_eps.shape
        if args.encoder_type == "Nature":
            encoder = NatureCNN(observation_shape[1], args)
        elif args.encoder_type == "Impala":
            encoder = ImpalaCNN(observation_shape[1], args)

        if args.weights_path == "None":
            if args.method not in probe_only_methods:
                sys.stderr.write("Probing without loading in encoder weights! Are sure you want to do that??")
        else:
            print("Print loading in encoder weights from probe of type {} from the following path: {}"
                  .format(args.method, args.weights_path))
            encoder.load_state_dict(torch.load(args.weights_path))
            encoder.eval()

    torch.set_num_threads(1)

    if args.method == 'majority':
        test_acc, test_f1score = majority_baseline(tr_labels, test_labels, wandb)

    else:
        test_acc, test_f1score = train_all_probes(encoder, tr_eps, val_eps,test_eps,  tr_labels, val_labels, test_labels, args, wandb.run.dir)

    acc_dict, f1_dict = postprocess_raw_metrics(test_acc, test_f1score)

    print("""In our paper, we report F1 scores and accuracies averaged across each category. 
          That is, we take a mean across all state variables in a category to get the average score for that category.
          Then we average all the category averages to get the final score that we report per game for each method. 
          These scores are called \'across_categories_avg_acc\' and \'across_categories_avg_f1\' respectively
          We do this to prevent categories with large number of state variables dominating the mean F1 score.
          """)


    print(test_acc, test_f1score)
    wandb.log(test_acc)
    wandb.log(test_f1score)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['probe']
    wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    run_probe(args)
