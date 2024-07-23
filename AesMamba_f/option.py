import argparse

def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument('--path_to_train_csv', dest='path_to_train_csv',
                        help='directory to train dataset',
                        default="/data/yuhao/dataset/PARA/annotation/PARA-GiaaTrain_w_class.csv",
                        # default="/data/yuhao/aesthetic_quality_evaluation/data/JAS/train.csv",
                        type=str)
    parser.add_argument('--path_to_test_csv', dest='path_to_test_csv',
                        help='directory to test dataset',
                        default='/data/yuhao/dataset/PARA/annotation/PARA-GiaaTest_w_class.csv',
                        type=str)
    parser.add_argument('--path_to_imgs', dest='path_to_imgs',
                        help='directory to images',
                        default="/data/yuhao/dataset/PARA/imgs/", type=str)
                        # default="/data/yuhao/dataset/JASDataset/dataset", type=str)

    # parser.add_argument('--path_to_model_weight', type=str, default='/data/yuhao/aesthetic_quality_evaluation/code/JAS_aesthetic/checkpoint/v1_best/best_acc.pth',
    #                     help='directory to pretrain models')

    parser.add_argument('--experiment_dir_name', type=str, default='.',
                        help='directory to project')

    parser.add_argument('--init_lr', type=int, default=1e-5, help='learning_rate'
                        )
    parser.add_argument('--num_epoch', type=int, default=100, help='epoch num for train'
                        )
    parser.add_argument('--batch_size', type=int, default=16, help='how many pictures to process one time'
                        )
    parser.add_argument('--num_workers', type=int, default=16, help ='num_workers',
                        )
    parser.add_argument('--gpu_id', type=str, default='3', help='which gpu to use')


    args = parser.parse_args()
    return args