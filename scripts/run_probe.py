from scripts.run_contrastive import train_encoder
from aari.probe import ProbeTrainer

import torch
from src.utils import get_argparser, train_encoder_methods, probe_only_methods
from src.encoders import NatureCNN, ImpalaCNN
import wandb
import sys
from src.majority import majority_baseline
from aari.episodes import get_episodes


def run_probe(args):
    device = torch.device("cuda:" + str(args.cuda_id) if torch.cuda.is_available() else "cpu")
    collect_mode = args.collect_mode if args.method != 'pretrained-rl-agent' else "pretrained_representations"
    tr_eps, val_eps, tr_labels, val_labels, test_eps, test_labels = get_episodes(args,device, collect_mode=collect_mode,
                                                                                 train_mode="probe")
    print("got episodes!")
    observation_shape = tr_eps[0][0].shape
    wandb.config.update(vars(args))

    if args.train_encoder and args.method in train_encoder_methods:
        print("Training encoder from scratch")
        encoder = train_encoder(args)
        encoder.probing = True
        encoder.eval()

    else:
        if args.encoder_type == "Nature":
            encoder = NatureCNN(observation_shape[0], args)
        elif args.encoder_type == "Impala":
            encoder = ImpalaCNN(observation_shape[0], args)

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
        trainer = ProbeTrainer(encoder,
                               wandb,
                               epochs=args.epochs,
                               sample_label=tr_labels[0][0],
                               lr=args.probe_lr,
                               batch_size=args.batch_size,
                               device=device,
                               patience=args.patience,
                               log=False)

        trainer.train(tr_eps, val_eps, tr_labels, val_labels)
        test_acc, test_f1score = trainer.test(test_eps, test_labels)

    print(test_acc)
    print(test_f1score)
    wandb.log(test_acc)
    wandb.log(test_f1score)


if __name__ == "__main__":
    parser = get_argparser()
    args = parser.parse_args()
    tags = ['probe']
    wandb.init(project=args.wandb_proj, entity="curl-atari", tags=tags)
    run_probe(args)
