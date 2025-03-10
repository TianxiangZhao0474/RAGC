from utils import *
from tqdm import tqdm
from torch import optim
from setup import setup_args
from model import sample_aware_network


if __name__ == '__main__':

    for dataset_name in ["cora"]:

        # setup hyper-parameter
        args = setup_args(dataset_name)

        # record results
        file = open("result.csv", "a+")
        print(args.dataset, file=file)
        print("ACC,   NMI,   ARI,   F1", file=file)
        file.close()
        acc_list = []
        nmi_list = []
        ari_list = []
        f1_list = []

        # ten runs with different random seeds
        for args.seed in range(args.runs):
            # record results

            # fix the random seed
            setup_seed(args.seed)

            # load graph data
            X, y, A, node_num, cluster_num = load_graph_data(dataset_name, show_details=False)

            # build sample aware network
            RAGC = sample_aware_network(
                input_dim=X.shape[1], hidden_dim=args.dims, act=args.activate, n_num=node_num)

            # apply the Hybrid_Collaborative_Augmentation
            X_aug = Hybrid_Collaborative_Augmentation(A, X, X, args.t_l, args.t_m)

            # test
            args.acc, args.nmi, args.ari, args.f1, y_hat, center = phi(X_aug, y, cluster_num)

            # adam optimizer
            optimizer = optim.Adam(RAGC.parameters(), lr=args.lr)

            # positive and negative sample pair index matrix
            mask = torch.ones([node_num * 2, node_num * 2]) - torch.eye(node_num * 2)

            # load data to device
            A, RAGC, X_aug, mask = map(lambda x: x.to(args.device), (A, RAGC, X_aug, mask))

            args.tao_l = args.tao

            # training
            for epoch in tqdm(range(400), desc="training..."):
                # train mode
                RAGC.train()

                # encoding
                Za, Zb, Ea, Eb = RAGC(X_aug, A)

                # calculate comprehensive similarity
                S = comprehensive_similarity(Za, Zb, Ea, Eb, RAGC.alpha)

                # calculate Sample Adaptive Differential Awareness contrastive loss
                loss = Sample_Adaptive_Differential_Awareness_infoNCE(S, mask, RAGC.pos_neg_weight, RAGC.pos_weight, node_num)

                # optimization
                loss.backward()
                optimizer.step()

                # update A_aug
                with torch.no_grad():
                    A_aug = (Za @ Zb.T + Ea @ Eb.T)
                    A_aug = (A_aug - A_aug.min()) / (A_aug.max() - A_aug.min())
                    A = A_aug * A

                # testing and update weights of sample pairs
                if epoch % 5 == 0:
                    # evaluation mode
                    RAGC.eval()

                    # encoding
                    Za, Zb, Ea, Eb = RAGC(X_aug, A)

                    # calculate comprehensive similarity
                    S = comprehensive_similarity(Za, Zb, Ea, Eb, RAGC.alpha)

                    # fusion and testing
                    Z = (Za + Zb) / 2
                    acc, nmi, ari, f1, P, center = phi(Z, y, cluster_num)

                    # high confidence samples
                    H, H_mat = high_confidence(Z, center)

                    # calculate new weight of sample pair
                    M, M_mat = pseudo_matrix(P, S, node_num)

                    # update weight
                    RAGC.pos_weight[H] = M[H].data
                    RAGC.pos_neg_weight[H_mat] = M_mat[H_mat].data

                    # Dynamic High Confidence Samples Selection
                    args.tao_l = max(args.tao * 0.1, args.tao_l * 0.95)

                    # recording
                    if acc >= args.acc:
                        args.acc, args.nmi, args.ari, args.f1 = acc, nmi, ari, f1

            print("Training complete")

            # record results
            file = open("result.csv", "a+")
            print("{:.2f}, {:.2f}, {:.2f}, {:.2f}".format(args.acc, args.nmi, args.ari, args.f1), file=file)
            file.close()
            acc_list.append(args.acc)
            nmi_list.append(args.nmi)
            ari_list.append(args.ari)
            f1_list.append(args.f1)

        # record results
        acc_list, nmi_list, ari_list, f1_list = map(lambda x: np.array(x), (acc_list, nmi_list, ari_list, f1_list))
        file = open("result.csv", "a+")
        print(f"tao: {args.tao}, beta: {args.beta}, gama: {args.gama}, t_l: {args.t_l}, t_m: {args.t_m}, lr: {args.lr}, dims: {args.dims}", file=file)
        print("{:.2f}, {:.2f}".format(acc_list.mean(), acc_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(nmi_list.mean(), nmi_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(ari_list.mean(), ari_list.std()), file=file)
        print("{:.2f}, {:.2f}".format(f1_list.mean(), f1_list.std()), file=file)
        file.close()
