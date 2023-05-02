from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../../../')
sys.path.append('../../../python_parser')
import warnings
warnings.filterwarnings("ignore")
import argparse
from tqdm import tqdm, trange
from run_parser import get_identifiers_ori


def generate(args):
    source_codes = []
    with open(args.train_data_file) as rf:
        for line in rf:
            source_codes.append(line.split(' <CODESPLIT> ')[0].strip().replace("\\n", "\n").replace('\"', '"'))
    with open(args.eval_data_file) as rf:
        for line in rf:
            source_codes.append(line.split(' <CODESPLIT> ')[0].strip().replace("\\n", "\n").replace('\"', '"'))
    results = set()
    for i in tqdm(source_codes):
        identifiers, code_tokens = get_identifiers_ori(i, args.language_type)
        for j in identifiers:
            results.add(j)
    results = [i+'\n' for i in results]
    print(len(results))
    open('../dataset/all_vars.txt', 'w').writelines(results)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.train_data_file = '../dataset/train.txt'
    if args.dataset_name == 'AuthorshipAttribution':
        args.language_type = 'python'
        args.eval_data_file = '../dataset/valid.txt'
    elif args.dataset_name == 'DefectPrediction':
        args.language_type = 'c'
        args.eval_data_file = '../dataset/valid.txt'
    elif args.dataset_name == 'FunctionalityClassification':
        args.language_type = 'c'
        args.eval_data_file = '../dataset/valid.txt'
    elif args.dataset_name == 'Cplusplus1000':
        args.language_type = 'c'
        args.eval_data_file = '../dataset/test.txt'
    elif args.dataset_name == 'Python800':
        args.language_type = 'python'
        args.eval_data_file = '../dataset/test.txt'
    elif args.dataset_name == 'Java250':
        args.language_type = 'java'
        args.eval_data_file = '../dataset/test.txt'
    generate(args)


if __name__ == "__main__":
    main()