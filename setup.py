from opt import args


def setup_args(dataset_name="cora"):
    args.dataset = dataset_name
    args.device = "cuda:0"
    args.acc = args.nmi = args.ari = args.f1 = 0

    if args.dataset == 'cora':
        args.t_l = 2
        args.t_m = 2
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.85
        args.beta = 0.9
        args.gama = 2

    elif args.dataset == 'citeseer':
        args.t_l = 3
        args.t_m = 1
        args.lr = 1e-3
        args.n_input = 500
        args.dims = 1500
        args.activate = 'sigmoid'
        args.tao = 0.9
        args.beta = 0.9
        args.gama = 1

    elif args.dataset == 'amap':
        args.t_l = 2
        args.t_m = 4
        args.lr = 5e-5
        args.n_input = -1
        args.dims = 1000
        args.activate = 'ident'
        args.tao = 0.8
        args.beta = 0.1
        args.gama = 2

    elif args.dataset == 'bat':
        args.t_l = 5
        args.t_m = 6
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.15
        args.beta = 0.9
        args.gama = 1

    elif args.dataset == 'eat':
        args.t_l = 6
        args.t_m = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.95
        args.beta = 0.7
        args.gama = 5

    elif args.dataset == 'uat':
        args.t_l = 6
        args.t_m = 6
        args.lr = 1e-4
        args.n_input = -1
        args.dims = 1000
        args.activate = 'sigmoid'
        args.tao = 0.1
        args.beta = 0.8
        args.gama = 5


    # other new datasets
    else:
        args.t_l = 3
        args.t_m = 3
        args.lr = 1e-3
        args.n_input = -1
        args.dims = 1500
        args.activate = 'ident'
        args.tao = 0.9
        args.beta = 0.9
        args.gama = 1

    print("---------------------")
    print("runs: {}".format(args.runs))
    print("dataset: {}".format(args.dataset))
    print("confidence: {}".format(args.tao))
    print("beta: {}".format(args.beta))
    print("gama: {}".format(args.gama))
    print("learning rate: {}".format(args.lr))
    print("---------------------")

    return args
