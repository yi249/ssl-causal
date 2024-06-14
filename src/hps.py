import argparse

HPARAMS_REGISTRY = {}


class Hparams:
    def update(self, dict):
        for k, v in dict.items():
            setattr(self, k, v)

cmmnist = Hparams()
cmmnist.lr = 1e-3
cmmnist.bs = 32
cmmnist.wd = 0.01
cmmnist.z_dim = 16
cmmnist.input_res = 32
cmmnist.input_channels = 3
cmmnist.pad = 4
cmmnist.enc_arch = "32b3d2,16b3d2,8b3d2,4b3d4,1b4"
cmmnist.dec_arch = "1b4,4b4,8b4,16b4,32b4"
cmmnist.widths = [16, 32, 64, 128, 256]
cmmnist.parents_x = ['fg_r', 'fg_g', 'fg_b', 'bg_r', 'bg_g', 'bg_b', 'thickness', 'intensity', 'digit']
cmmnist.parent_names = ["fgcol", "bgcol", "thickness", "intensity", "digit"]
cmmnist.concat_pa = False
cmmnist.context_norm = "[-1,1]"
cmmnist.context_dim = 18
HPARAMS_REGISTRY["cmmnist"] = cmmnist


mimic192 = Hparams()
mimic192.lr = 1e-3
mimic192.bs = 6
mimic192.wd = 0.1
mimic192.z_dim = 16
mimic192.input_res = 192
mimic192.pad = 9
mimic192.enc_arch = "192b1d2,96b3d2,48b7d2,24b11d2,12b7d2,6b3d6,1b2"
mimic192.dec_arch = "1b2,6b4,12b8,24b12,48b8,96b4,192b2"
mimic192.widths = [32, 64, 96, 128, 160, 192, 512]
mimic192.parents_x = ['finding', 'age', 'sex', 'race']
mimic192.parent_names = ['finding', 'age', 'sex', 'race']
mimic192.concat_pa = False
mimic192.context_dim = 6
HPARAMS_REGISTRY["mimic192"] = mimic192

def range_limited_float_type(arg):
    """ Type function for argparse - a float within some predefined bounds """
    try:
        f = float(arg)
    except ValueError:
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < 0 or f > 1:
        raise argparse.ArgumentTypeError("Argument must be < " + str(0) + "and > " + str(1))
    return f

def setup_hparams(parser: argparse.ArgumentParser) -> Hparams:
    hparams = Hparams()
    args = parser.parse_known_args()[0]
    valid_args = set(args.__dict__.keys())
    hparams_dict = HPARAMS_REGISTRY[args.hps].__dict__
    for k in hparams_dict.keys():
        if k not in valid_args:
            raise ValueError(f"{k} not in default args")
    parser.set_defaults(**hparams_dict)
    hparams.update(parser.parse_known_args()[0].__dict__)
    return hparams


def add_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--labelled", help="Proportion of data that is labelled.", type=str, default="")
    parser.add_argument("--exp_name", help="Experiment name.", type=str, default="")
    parser.add_argument(
        "--data_dir", help="Data directory to load form.", type=str, default=""
    )
    parser.add_argument("--random", help="Use random missing data.", action="store_true", default=False)
    parser.add_argument("--hps", help="hyperparam set.", type=str, default="ukbb64")
    parser.add_argument(
        "--resume", help="Path to load checkpoint.", type=str, default=""
    )
    parser.add_argument("--seed", help="Set random seed.", type=int, default=7)
    parser.add_argument("--wandb", help="Whether to use wandb", action="store_true", default=False)
    parser.add_argument(
        "--deterministic",
        help="Toggle cudNN determinism.",
        action="store_true",
        default=False,
    )
    parser.add_argument("--device", help="Device to use", type=str, default="cuda:1")
    # training
    parser.add_argument("--epochs", help="Training epochs.", type=int, default=1000)
    parser.add_argument(
"--scm_thresh",
        help="When to start using scm outputs",
        nargs="+",
        type=int,
        default=[0, 5],
    )
    parser.add_argument(
"--reg_thresh",
        help="When to start counterfactual regularisation",
        nargs="+",
        type=int,
        default=[5, 10],
    )
    parser.add_argument("--bs", help="Batch size.", type=int, default=32)
    parser.add_argument("--lr", help="Learning rate.", type=float, default=1e-3)
    parser.add_argument(
        "--lr_warmup_steps", help="lr warmup steps.", type=int, default=100
    )
    parser.add_argument("--wd", help="Weight decay penalty.", type=float, default=0.01)
    parser.add_argument(
        "--betas",
        help="Adam beta parameters.",
        nargs="+",
        type=float,
        default=[0.9, 0.9],
    )
    parser.add_argument("--rw", help="Regularisation weight", type=float, default=0.1)
    parser.add_argument("--zrw", help="Z regularisation weight,", type=float, default=0.1)
    parser.add_argument(
        "--ema_rate", help="Exp. moving avg. model rate.", type=float, default=0.999
    )
    parser.add_argument(
        "--input_res", help="Input image crop resol ution.", type=int, default=64
    )
    parser.add_argument(
        "--input_channels", help="Input image num channels.", type=int, default=1
    )
    parser.add_argument("--pad", help="Input padding.", type=int, default=3)
    parser.add_argument(
        "--hflip", help="Horizontal flip prob.", type=float, default=0.5
    )
    parser.add_argument(
        "--grad_clip", help="Gradient clipping value.", type=float, default=350
    )
    parser.add_argument(
        "--grad_skip", help="Skip update grad norm threshold.", type=float, default=1000
    )
    parser.add_argument(
        "--accu_steps", help="Gradient accumulation steps.", type=int, default=1
    )
    parser.add_argument(
        "--beta", help="Max KL beta penalty weight.", type=float, default=1.0
    )
    parser.add_argument(
        "--cw", help="Classifier weight.", type=float, default=1.0
    )
    parser.add_argument(
        "--beta_warmup_steps", help="KL beta penalty warmup steps.", type=int, default=0
    )
    parser.add_argument(
        "--kl_free_bits", help="KL min free bits constraint.", type=float, default=0.0
    )
    parser.add_argument(
        "--viz_freq", help="Steps per visualisation.", type=int, default=10000
    )
    parser.add_argument(
        "--eval_freq", help="Train epochs per validation.", type=int, default=5
    )
    # model
    parser.add_argument(
        "--vae",
        help="VAE model: simple/hierarchical.",
        type=str,
        default="hierarchical",
    )
    parser.add_argument(
        "--enc_arch",
        help="Encoder architecture config.",
        type=str,
        default="64b1d2,32b1d2,16b1d2,8b1d8,1b2",
    )
    parser.add_argument(
        "--dec_arch",
        help="Decoder architecture config.",
        type=str,
        default="1b2,8b2,16b2,32b2,64b2",
    )
    parser.add_argument(
        "--cond_prior",
        help="Use a conditional prior.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--widths",
        help="Number of channels.",
        nargs="+",
        type=int,
        default=[16, 32, 48, 64, 128],
    )
    parser.add_argument(
        "--bottleneck", help="Bottleneck width factor.", type=int, default=4
    )
    parser.add_argument(
        "--z_dim", help="Numver of latent channel dims.", type=int, default=16
    )
    parser.add_argument(
        "--z_max_res",
        help="Max resolution of stochastic z layers.",
        type=int,
        default=192,
    )
    parser.add_argument(
        "--bias_max_res",
        help="Learned bias param max resolution.",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--x_like",
        help="x likelihood: {fixed/shared/diag}_{gauss/dgauss}.",
        type=str,
        default="diag_dgauss",
    )
    parser.add_argument(
        "--std_init",
        help="Initial std for x scale. 0 is random.",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--parents_x",
        help="Parents of x to condition on.",
        nargs="+",
        default=['fg_r', 'fg_g', 'fg_b', 'bg_r', 'bg_g', 'bg_b', 'thickness', 'intensity', 'digit'],
    )
    parser.add_argument(
        "--parent_names",
        help="Names of parent variables",
        nargs="+",
        default=["fgcol", "bgcol", "thickness", "intensity", "digit"],
    )
    parser.add_argument(
        "--concat_pa",
        help="Whether to concatenate parents_x.",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--context_dim",
        help="Num context variables conditioned on.",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--context_norm",
        help='Conditioning normalisation {"[-1,1]"/"[0,1]"/log_standard}.',
        type=str,
        default="log_standard",
    )
    parser.add_argument(
        "--q_correction",
        help="Use posterior correction.",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "--flow_widths",
        help="Cond flow fc network width per layer.",
        nargs="+",
        type=int,
        default=[32, 32],
    )
    parser.add_argument(
        "--std_fixed", help="Fix aux dist std value (0 is off).", type=float, default=0
    )
    return parser
