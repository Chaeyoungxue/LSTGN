import argparse

def parse_args():
    # 创建一个 ArgumentParser 对象，用于解析命令行参数，描述信息为 'LSTAN'
    parser = argparse.ArgumentParser(description='LSTAN')

    # 添加一个名为 --rgb_list 的命令行参数，默认值是 'ucf_x3d_train.txt'，用于指定 RGB 特征列表文件
    parser.add_argument('--rgb_list', default='ucf_x3d_train.txt', help='list of rgb features ')

    # 添加一个名为 --test_rgb_list 的命令行参数，默认值是 'ucf_x3d_test.txt'，用于指定测试 RGB 特征列表文件
    parser.add_argument('--test_rgb_list', default='ucf_x3d_test.txt', help='list of test rgb features ')

    # 添加一个名为 --comment 的命令行参数，默认值是 'tiny'，用于为训练的检查点名称添加注释
    parser.add_argument('--comment', default='LSTAN', help='comment for the ckpt name of the training')
    # 添加一个名为 --dropout_rate 的命令行参数，类型为浮点数，默认值是 0.4，用于指定 Dropout 率
    parser.add_argument('--dropout_rate', type=float, default=0.4, help='dropout rate')

    # 添加一个名为 --attn_dropout_rate 的命令行参数，类型为浮点数，默认值是 0.1，用于指定注意力机制的 Dropout 率
    parser.add_argument('--attn_dropout_rate', type=float, default=0.1, help='attention dropout rate')

    # 添加一个名为 --lr 的命令行参数，类型为字符串，默认值是 2e-4，用于指定学习率
    parser.add_argument('--lr', type=str, default=1e-4, help='learning rates for steps default:2e-4')

    # 添加一个名为 --batch_size 的命令行参数，类型为整数，默认值是 16，用于指定每个数据批次中的实例数量
    parser.add_argument('--batch_size', type=int, default=16, help='number of instances in a batch of data (default: 16)')

    # 添加一个名为 --model_name 的命令行参数，默认值是 'model'，用于指定保存模型的名称
    parser.add_argument('--model_name', default='model', help='name to save model')

    # 添加一个名为 --pretrained_ckpt 的命令行参数，默认值是 None，用于指定预训练模型的检查点文件
    parser.add_argument('--pretrained_ckpt', default=None, help='ckpt for pretrained model')

    # 添加一个名为 --max_epoch 的命令行参数，类型为整数，默认值是 30，用于指定最大训练轮数
    parser.add_argument('--max_epoch', type=int, default=30, help='maximum iteration to train (default: 10)')

    # 添加一个名为 --warmup 的命令行参数，类型为整数，默认值是 1，用于指定热身训练的轮数
    parser.add_argument('--warmup', type=int, default=1, help='number of warmup epochs')

    args = parser.parse_args()
    return args
