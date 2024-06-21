import argparse

def str2bool(str):
    return True if str.lower() == 'true' else False




def parse_args():
    parser = argparse.ArgumentParser(description="KGIN")

    # ===== dataset ===== #
    parser.add_argument("--dataset", nargs="?", default="alibaba-fashion",
                        help="Choose a dataset:[last-fm,amazon-book,alibaba]")
    parser.add_argument(
        "--data_path", nargs="?", default="data/", help="Input data path."
    )

    parser.add_argument('--load_rules', type=str2bool,
                        default=True, help='whether to load outfit rules')

    parser.add_argument('--which_rule', type=int,
                        default=2, help='load which outfit rules')
    # ===== train ===== #
    parser.add_argument('--epoch', type=int, default=150,
                        help='number of epochs')
    parser.add_argument('--batch_size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--test_batch_size', type=int,
                        default=1024, help='batch size')
    parser.add_argument('--dim', type=int, default=64, help='embedding size')
    parser.add_argument('--l2', type=float, default=1e-5,
                        help='l2 regularization weight')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--sim_regularity', type=float, default=1e-4,
                        help='regularization weight for latent factor')
    parser.add_argument("--inverse_r", type=str2bool, default=True,
                        help="consider inverse relation or not")
    parser.add_argument("--node_dropout", type=str2bool,
                        default=True, help="consider node dropout or not")
    parser.add_argument("--node_dropout_rate", type=float,
                        default=0.5, help="ratio of node dropout")
    parser.add_argument("--mess_dropout", type=str2bool,
                        default=True, help="consider message dropout or not")
    parser.add_argument("--mess_dropout_rate", type=float,
                        default=0.1, help="ratio of node dropout")
    parser.add_argument("--batch_test_flag", type=str2bool,
                        default=True, help="use gpu or not")
    parser.add_argument("--channel", type=int, default=64,
                        help="hidden channels for model")
    parser.add_argument("--cuda", type=str2bool, default=True,
                        help="use gpu or not")
    parser.add_argument("--gpu_id", type=int, default=0, help="gpu id")
    parser.add_argument("--edge_threshold", type=int, default=64,
                        help="edge threshold to filter knowledge graph")
    parser.add_argument(
        "--adj_epoch", type=int, default=1, help="build adj matrix per _ epoch"
    )
    parser.add_argument(
        "--in_channel", type=str, default="[64, 32]", help="input channels for gcn"
    )
    parser.add_argument(
        "--out_channel", type=str, default="[32, 64]", help="output channels for gcn"
    )
    parser.add_argument(
        "--description", type=str, default="none", help="name the result file"
    )
    parser.add_argument(
        "--pretrain_s",
        type=str2bool,
        default=False,
        help="load pretrained sampler data or not",
    )
    parser.add_argument(
        "--pretrain_r", type=str2bool, default=False, help="use pretrained model or not"
    )
    parser.add_argument(
        "--freeze_s",
        type=str2bool,
        default=False,
        help="freeze parameters of recommender or not",
    )
    parser.add_argument(
        "--k_step", type=int, default=1, help="k step from current positive items"
    )
    parser.add_argument(
        "--num_sample", type=int, default=32, help="number fo samples from gcn"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="gamma for reward accumulation"
    )

    parser.add_argument(
        '--Ks', nargs='?', default='[20, 40, 60, 80, 100]', help='Output sizes of every layer')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument("--n_factors", type=int, default=4,
                        help="number of latent factor for user favour")
    parser.add_argument("--ind", type=str, default='distance',
                        help="Independence modeling: mi, distance, cosine")

    # ===== relation context ===== #
    parser.add_argument('--context_hops', type=int,
                        default=3, help='number of context hops')

    # ===== save model ===== #
    parser.add_argument("--save", type=str2bool, default=True,
                        help="save model or not")
    parser.add_argument("--out_dir", type=str,
                        default="./weights/", help="output directory for model")
    # ------------------------- experimental settings specific for testing0 ---------------------------------------------
    parser.add_argument(
        "--rank", nargs="?", default="[20, 40, 60, 80, 100]", help="evaluate K list"
    )
    parser.add_argument("--flag_step", type=int,
                        default=10, help="early stop steps")

    return parser.parse_args()
