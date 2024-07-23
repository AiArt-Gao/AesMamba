import argparse


def init():
    parser = argparse.ArgumentParser(description="PyTorch")
    parser.add_argument('--path_to_ava_txt', type=str, default="/data/yuhao/datasets/AVA_dataset/mat_file/AVA.txt",
                        help='directory to csv_folder')

    parser.add_argument('--path_to_images', type=str, default='/data/yuhao/dataset/AVADataset',
                        help='directory to images')

    parser.add_argument('--path_to_comments', type=str, default='/data/yuhao/dataset/AVA_Comment_Dataset',
                        help='directory to images')

    parser.add_argument('--path_to_save_csv', type=str, default="/data/yuhao/aesthetic_quality_assessment/data/AVA",
                        help='directory to csv_folder')

    parser.add_argument('--experiment_dir_name', type=str, default='.',
                        help='directory to project')

    parser.add_argument('--path_to_model_weight', type=str,
                        # default='/data/yuhao/pretrain_model/poolformer/poolformer_s12.pth.tar',
                        default='/data/yuhao/aesthetic_quality_assessment/code_/AVA_comment/checkpoint/clip_v5/best_srcc.pth',
                        help='directory to pretrain models')

    parser.add_argument('--init_lr', type=int, default=1e-5, help='learning_rate'
                        )
    parser.add_argument('--num_epoch', type=int, default=50, help='epoch num for train'
                        )
    parser.add_argument('--batch_size', type=int, default=64, help='how many pictures to process one time'
                        )
    parser.add_argument('--num_workers', type=int, default=16, help='num_workers',
                        )
    parser.add_argument('--gpu_id', type=str, default='1', help='which gpu to use')

    args = parser.parse_args()
    return args
