from scripts.run_contrastive import train_encoder
from aari.probe import ProbeTrainer


import numpy as np
import torch
from aari.envs import make_vec_envs
from src.utils import get_argparser, appendabledict, train_encoder_methods
from src.encoders import NatureCNN, ImpalaCNN
import wandb
import sys
from src.majority import majority_baseline
from aari.episodes import get_episodes



def run_probe(encoder, args, device, seed):
    collect_mode = args.collect_mode if args.method != 'pretrained-rl-agent' else "pretrained_representations"

    tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels = get_episodes(args,device, collect_mode=collect_mode, train_mode="probe" ,seed=seed)
    print("got episodes!")

    if args.method == 'majority':
        return majority_baseline(tr_labels, test_labels, wandb)

    trainer = ProbeTrainer(encoder,
                           wandb,
                           epochs=args.epochs,
                           sample_label=tr_labels[0][0],
                           lr=args.lr,
                           batch_size=args.batch_size,
                           device=device,
                           patience=args.patience,
                           log=False)

    trainer.train(tr_eps, val_eps, tr_labels, val_labels)
    test_acc, test_f1score = trainer.test(test_eps, test_labels)

    return test_acc, test_f1score


def main(args):
    # dummy env
    env = make_vec_envs(args, 1)
    wandb.config.update(vars(args))

    if args.train_encoder and args.method in train_encoder_methods:
        print("Training encoder from scratch")
        encoder = train_encoder(args)
        encoder.probing = True
        encoder.eval()

    else:
        if args.encoder_type == "Nature":
            encoder = NatureCNN(env.observation_space.shape[0], args)
        elif args.encoder_type == "Impala":
            encoder = ImpalaCNN(env.observation_space.shape[0], args)

        if args.method == "random_cnn":
            print("Random CNN, so not loading in encoder weights!")
        if args.method == "majority":
            print("Majority baseline!")
        elif args.method == "supervised":
            print("Fully supervised, so starting from random encoder weights!")
        elif args.method == "pretrained-rl-agent":
            print("Representation from pretrained rl agent, so we don't need an encoder!")
        else:
            if args.weights_path == "None":
                sys.stderr.write("Probing without loading in encoder weights! Are sure you want to do that??")
            else:
                print("Print loading in encoder weights from probe of type {} from the following path: {}"
                      .format(args.method, args.weights_path))
                encoder.load_state_dict(torch.load(args.weights_path))
                encoder.eval()

    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    env.close()

    # encoder.to(device)
    torch.set_num_threads(1)

    all_runs_test_f1 = appendabledict()
    all_runs_test_acc = appendabledict()
    for i, seed in enumerate(range(args.seed, args.seed + args.num_runs)):
        print("Run number {} of {}".format(i + 1, args.num_runs))
        test_acc, f1score = run_probe(encoder, args, device, seed=1)
        all_runs_test_f1.append_update(f1score)
        all_runs_test_acc.append_update(test_acc)

    mean_acc_dict = {"mean_" + k: np.mean(v) for k, v in all_runs_test_acc.items()}
    var_acc_dict = {"var_" + k: np.var(v) for k, v in all_runs_test_acc.items()}
    mean_f1_dict = {"mean_" + k: np.mean(v) for k, v in all_runs_test_f1.items()}
    var_f1_dict = {"var_" + k: np.var(v) for k, v in all_runs_test_f1.items()}
    print(mean_acc_dict)
    print(var_acc_dict)
    wandb.log(mean_acc_dict)
    wandb.log(var_acc_dict)
    print(mean_f1_dict)
    print(var_f1_dict)
    wandb.log(mean_f1_dict)
    wandb.log(var_f1_dict)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['probe']
    wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    main(args)
