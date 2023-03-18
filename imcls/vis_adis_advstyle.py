import argparse
import torch
import os.path as osp
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import ipdb

def normalize(feature):
    norm = np.sqrt((feature**2).sum(1, keepdims=True))
    return feature / (norm + 1e-12)


import numpy as np
from sklearn import svm

def proxy_a_distance(source_X, target_X, verbose=False):
    """
    Compute the Proxy-A-Distance of a source/target representation
    """
    nb_source = np.shape(source_X)[0]
    nb_target = np.shape(target_X)[0]

    if verbose:
        print('PAD on', (nb_source, nb_target), 'examples')

    # C_list = np.logspace(0, 0, 10)
    C_list = np.logspace(-5, 3, 8)

    half_source, half_target = int(nb_source/2), int(nb_target/2)
    train_X = np.vstack((source_X[0:half_source, :], target_X[0:half_target, :]))
    train_Y = np.hstack((np.zeros(half_source, dtype=int), np.ones(half_target, dtype=int)))

    test_X = np.vstack((source_X[half_source:, :], target_X[half_target:, :]))
    test_Y = np.hstack((np.zeros(nb_source - half_source, dtype=int), np.ones(nb_target - half_target, dtype=int)))

    best_risk = 1.0
    for C in C_list:
        clf = svm.SVC(C=C, kernel='linear', verbose=False)
        clf.fit(train_X, train_Y)

        train_risk = np.mean(clf.predict(train_X) != train_Y)
        test_risk = np.mean(clf.predict(test_X) != test_Y)

        if verbose:
            print('[ PAD C = %f ] train risk: %f  test risk: %f' % (C, train_risk, test_risk))

        if test_risk > .5:
            test_risk = 1. - test_risk

        best_risk = min(best_risk, test_risk)

    return 2 * (1. - 2 * best_risk)


def plot_histogram(x, y, src, name):
    # x: M; y: N

    import matplotlib.pyplot as plt
    import numpy as np

    bins = np.linspace(0.18, 0.7, 200)

    plt.hist(x, bins=100, alpha=0.5, label='Training')
    plt.hist(y, bins=100, alpha=0.5, label='Test')
    plt.legend(loc='upper right')
    plt.savefig(osp.join(src, name + 'mean_his_final_cate.pdf'), bbox_inches='tight')
    plt.close()
    # plt.show()

    # plt.hist(x, density=True, bins=30)  # density=False would make counts
    # plt.ylabel('Probability')
    # plt.xlabel('Data');

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src', type=str, default='/home/yabin/syn_project/mixstyle-release-master/imcls/aaaivisual/pacs/Vanilla2_singles/resnet50/random/art_painting/seed2/', help='path to source file')
    parser.add_argument('--dst', type=str, default='', help='destination directory')
    parser.add_argument('--method', type=str, default='tsne', help='tnse, pca or none')
    args = parser.parse_args()

    resnet = '/home/yabin/syn_project/mixstyle-release-master/imcls/aaaivisual/pacs/Vanilla2_singles/resnet50/random/art_painting/seed2/'

    dsu = '/home/yabin/syn_project/mixstyle-release-master/imcls/aaaivisual/pacs/Vanilla2_singles/resnet50_dsus_lc01234/random/art_painting/seed1/'

    advstyle = '/home/yabin/syn_project/mixstyle-release-master/imcls/aaaivisual/pacs/Vanilla2_singles/resnet50_advs_lc01234_advw10.0_mixw1.0/random/art_painting/seed1/'

    list = [resnet, dsu, advstyle]
    name_list = ['resnet', 'dsu', 'advstyle']
    item = 'adis'
    # item = 'adis'
    # item = 'hist'

    for i in range(3):
        args.src = list[i]
        name = name_list[i]

        training_mean = args.src + 'embed_mean_train.pt'
        training_var = args.src + 'embed_var_train.pt'
        test_mean = args.src + 'embed_mean_test.pt'
        test_var = args.src + 'embed_var_test.pt'
        training_mean = torch.load(training_mean)
        training_var = torch.load(training_var)
        test_mean = torch.load(test_mean)
        test_var = torch.load(test_var)

        # print(test_mean['domain'])  ## 0: cartoon; 1: photo; 2:sketch
        # print(training_mean['domain'])

        # ipdb.set_trace()
        print(name)
        ######################################################### 计算 A-dis， 但是结果和我预期的正好相反。。。不知道为啥？
        if item == 'adis':
            for domain in range(3):
                test_mean_one_domain = test_mean['embed'][[test_mean['domain'] == domain]]
                dis = proxy_a_distance(training_mean['embed'], test_mean_one_domain)  # source: N*K, target: M*K
                print(domain, dis)
        ############################# mean, dis, 分domain 计算的。
        # 按A-dis 来说，我的方法的结果的A-dis 是最大的，也就是training 与 test 的差异是最大的... 可能是我的方法使用了更大的style space, 其是包含target 的，但是
        # advstyle: 2.0, 2.0, 2.0      | 2.0
        # DSU:      1.97, 1.91, 2.0    |
        # baseline: 1.93, 1.85, 2.0    | 1.96
        #########################################################


        ######################################################### 画均值直方图， 为每个类，每个domain 单独做。
        if item == 'hist':
            print('ploting the statistics historgram.')
            for domain in range(3):
                for cate in range(7):
                    test_mean_one_domain = test_mean['embed'][[test_mean['domain'] == 0]]
                    test_label_one_domain = test_mean['label'][[test_mean['domain'] == 0]]
                    name = 'D' + str(domain) + 'C' + str(cate)
                    plot_histogram(training_mean['embed'][training_mean['label'] == cate].mean(1), test_mean_one_domain[test_label_one_domain == cate].mean(1),  args.src, name)  # source: N*K, target: M*K
        ############################# 把mean 的均值画出来，发现
        #
        #########################################################




        # if not args.dst:
        #     args.dst = osp.dirname(args.src)
        #
        # print('Loading file from "{}"'.format(args.src))
        # file = torch.load(args.src)

        # embed = file['embed']
        # domain = file['domain']
        # dnames = file['dnames']
        # cate = file['label']
        if item == 'tsne':
            embed_train = training_mean['embed']
            train_num = embed_train.shape[0]

            for domain in range(3):
                print('processing domain', domain)
                embed_test = test_mean['embed'][test_mean['domain'] == domain]
                embed = np.concatenate([embed_train, embed_test], axis=0)
                figname = 'embed_mean_var_concat_final_all_test' + str(domain) + '.pdf'

                # embed = embed[cate==1] ## only the dog feature 1 is good
                # domain = domain[cate==1]


                #dim = embed.shape[1] // 2
                #embed = embed[:, dim:]

                #domain = file['label']
                #dnames = ['dog', 'elephant', 'giraffe', 'guitar', 'horse', 'house', 'person']

                # nd_src = len(dnames)
                # embed = normalize(embed)

                print('Loaded features with shape {}'.format(embed.shape))
                #
                # embed2d_path = osp.join(args.dst, 'embed2d_' + args.method + '.pt')

                # if osp.exists(embed2d_path):
                #     embed2d = torch.load(embed2d_path)
                #     print('Loaded embed2d from "{}"'.format(embed2d_path))
                #
                # else:
                if args.method == 'tsne':
                    print('Dimension reduction with t-SNE (dim=2) ...')
                    tsne = TSNE(
                        n_components=2, metric='euclidean', verbose=1,
                        perplexity=50, n_iter=1000, learning_rate=200.
                    )
                    embed2d = tsne.fit_transform(embed)

                    # torch.save(embed2d, embed2d_path)
                    # print('Saved embed2d to "{}"'.format(embed2d_path))

                # elif args.method == 'pca':
                #     print('Dimension reduction with PCA (dim=2) ...')
                #     pca = PCA(n_components=2)
                #     embed2d = pca.fit_transform(embed)
                #
                #     torch.save(embed2d, embed2d_path)
                #     print('Saved embed2d to "{}"'.format(embed2d_path))
                #
                # elif args.method == 'none':
                #     # the original embedding is 2-D
                #     embed2d = embed

                # avai_domains = list(set(domain.tolist()))
                # avai_domains.sort()

                print('Plotting ...')

                SIZE = 3
                COLORS = ['r', 'b', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
                LEGEND_MS = 3

                fig, ax = plt.subplots()

                for d in range(2):
                    d = int(d)
                    # e = embed2d[domain == d]
                    if d == 0:
                        e = embed2d[:train_num, :]
                        label = 'Source'
                    else:
                        e = embed2d[train_num:, :]
                        label = 'Target'

                    # cate_d = cate[domain == d]
                    # e = e[cate_d==0]
                    #
                    # """
                    # label = '$D_{}$'.format(str(d + 1))
                    # if d < nd_src:
                    #     label += ' ($\mathcal{S}$)'
                    # else:
                    #     label += ' ($\mathcal{N}$)'
                    # """
                    # label = dnames[d]
                    # print(label)

                    ax.scatter(
                        e[:, 0],
                        e[:, 1],
                        s=SIZE,
                        c=COLORS[d],
                        edgecolors='none',
                        label=label,
                        alpha=1,
                        rasterized=False
                    )

                    # ##################### plot the feature histogram of each domain
                    # e = embed[domain == d]  ## N * d
                    # ee = e.ravel()
                    # fig, ax = plt.subplots()
                    # plt.hist(ee, color=，bins=100)
                    # plt.xlabel('Values', fontsize=16)
                    # plt.ylabel('Numbers', fontsize=16)
                    # ax.set_xlim(-3, 6)
                    # ax.ticklabel_format(style='sci', scilimits=(-1, 2), axis='y')
                    # figname = label + '_feature_dist.pdf'
                    # plt.savefig(osp.join(args.dst, figname), bbox_inches='tight')
                    # plt.close()

                ax.legend(loc='upper left', fontsize=10, markerscale=LEGEND_MS)
                ax.legend(fontsize=14, markerscale=LEGEND_MS)
                ax.set_xticks([])
                ax.set_yticks([])
                # ax.set_xlim(-3, 10)
                #ax.set_ylim(-LIM, LIM)

                fig.savefig(osp.join(args.src, figname), bbox_inches='tight')
                plt.close()


if __name__ == '__main__':
    main()
