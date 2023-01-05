import argparse

def args_parser():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--data_distribution', default=1, type=int)
    parser.add_argument('--local_update_step_number', default=5, type=int)
    parser.add_argument('--momentum', default=0.9, type=float)  # 0.9
    parser.add_argument('--weight_decay', default=1e-4, type=float)  # 1e-4
    parser.add_argument('--seed', default=0, type=int)

    parser.add_argument('--workers', default=0, type=int)
    parser.add_argument('--epochs', default=10000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--lr', default=0.5, type=int)
    parser.add_argument('--half', default=0, type=int)

    args = parser.parse_args()

    return args