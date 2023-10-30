from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../../')
sys.path.append('../../python_parser')
import warnings
warnings.filterwarnings("ignore")
import os
import re
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import CodeBERT, GraphCodeBERT, CodeT5
from run import CodeBertTextDataset, GraphCodeBertTextDataset, CodeT5TextDataset
from run_parser import get_identifiers_ori, get_example, remove_comments_and_docstrings
from transformers import (RobertaModel, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, T5Config,
                          T5ForConditionalGeneration)


MODEL_CLASSES = {
    'codebert_roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def generate(args, model, tokenizer, mlm_tokenizer, data_file, method):
    source_codes = []
    with open(data_file) as rf:
        for line in rf:
            source_codes.append(line.split(' <CODESPLIT> ')[0].strip().replace("\\n", "\n").replace('\"', '"'))

    if args.model_name == 'codebert':
        dataset = CodeBertTextDataset(tokenizer, args, data_file)
    elif args.model_name == 'graphcodebert':
        dataset = GraphCodeBertTextDataset(tokenizer, args, data_file)
    elif args.model_name == 'codet5':
        dataset = CodeT5TextDataset(tokenizer, args, data_file)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0)
    model = torch.nn.DataParallel(model)
    model.eval()
    results = []
    for batch_i, batch in enumerate(tqdm(dataloader)):
        if args.model_name == 'codebert':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit, _ = model(inputs, labels)
        elif args.model_name == 'graphcodebert':
            inputs_ids = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            labels = batch[3].to(args.device)
            with torch.no_grad():
                lm_loss, logit, _ = model(inputs_ids, attn_mask, position_idx, labels)
        elif args.model_name == 'codet5':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                lm_loss, logit, _ = model(inputs, labels)
        label = labels.cpu().numpy()[0]
        label_pred = np.argmax(logit.cpu().numpy()[0])
        if method == 'train' and label_pred == label:
            ori_code_temp = source_codes[batch_i]
            try:
                code_temp = remove_comments_and_docstrings(ori_code_temp, args.language_type).strip()
            except:
                code_temp = ori_code_temp.strip()
            code_temp = re.sub(' +', ' ', code_temp)
            t = mlm_tokenizer.tokenize(code_temp)[:args.block_size]
            code_temp = mlm_tokenizer.convert_tokens_to_string(t)
            identifiers, code_tokens = get_identifiers_ori(code_temp, args.language_type)
            for iden in identifiers:
                results.append(ori_code_temp.replace('\n', '\\n').replace("\t", "\\t").replace('"', '\"') + ' <CODESPLIT> ' +
                               get_example(code_temp, iden, '<mask>', args.language_type).replace('\n', '\\n').replace("\t", "\\t").replace('"', '\"') + ' <CODESPLIT> ' +
                               iden + ' <CODESPLIT> ' + str(label)+ ' <CODESPLIT> ' + str(label_pred)+'\n')
        elif method == 'test' and label_pred != label:
            ori_code_temp = source_codes[batch_i]
            try:
                code_temp = remove_comments_and_docstrings(ori_code_temp, args.language_type).strip()
            except:
                code_temp = ori_code_temp.strip()
            code_temp = re.sub(' +', ' ', code_temp)
            t = mlm_tokenizer.tokenize(code_temp)[:args.block_size-2]
            code_temp = mlm_tokenizer.convert_tokens_to_string(t)
            results.append(ori_code_temp.replace('\n', '\\n').replace("\t", "\\t").replace('"', '\"')+' <CODESPLIT> '+
                           code_temp.replace('\n', '\\n').replace("\t","\\t").replace('"', '\"')+' <CODESPLIT> '+
                           '' + ' <CODESPLIT> ' + str(label) + ' <CODESPLIT> ' + str(label_pred)+'\n')
    open('../dataset/mlm_%s_%s.txt'%(args.model_name, method), 'w').writelines(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=True, help="model name.")
    args = parser.parse_args()
    args.output_dir = './saved_models/'
    args.train_data_file = '../dataset/train.txt'
    args.block_size = 512
    args.seed = 123456
    args.batch_size = 1

    if args.model_name == 'codebert':
        args.model_type = 'codebert_roberta'
        args.config_name = '/workspace/Attack/microsoft/codebert-base'
        args.model_name_or_path = '/workspace/Attack/microsoft/codebert-base'
        args.tokenizer_name = '/workspace/Attack/roberta-base'
    elif args.model_name == 'graphcodebert':
        args.model_type = 'graphcodebert_roberta'
        args.config_name = '/workspace/Attack/microsoft/graphcodebert-base'
        args.model_name_or_path = '/workspace/Attack/microsoft/graphcodebert-base'
        args.tokenizer_name = '/workspace/Attack/microsoft/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 64
    elif args.model_name == 'codet5':
        args.model_type = 'codet5'
        args.config_name = '/workspace/Attack/Salesforce/codet5-base-multi-sum'
        args.model_name_or_path = '/workspace/Attack/Salesforce/codet5-base-multi-sum'
        args.tokenizer_name = '/workspace/Attack/Salesforce/codet5-base-multi-sum'

    if args.dataset_name == 'AuthorshipAttribution':
        args.number_labels = 66
        args.language_type = 'python'
        args.eval_data_file = '../dataset/valid.txt'
    elif args.dataset_name == 'DefectPrediction':
        args.number_labels = 4
        args.language_type = 'c'
        args.eval_data_file = '../dataset/valid.txt'
    elif args.dataset_name == 'FunctionalityClassification':
        args.number_labels = 104
        args.language_type = 'c'
        args.eval_data_file = '../dataset/valid.txt'
    elif args.dataset_name == 'Cplusplus1000':
        args.number_labels = 1000
        args.language_type = 'c'
        args.eval_data_file = '../dataset/test.txt'
    elif args.dataset_name == 'Python800':
        args.number_labels = 800
        args.language_type = 'python'
        args.eval_data_file = '../dataset/test.txt'
    elif args.dataset_name == 'Java250':
        args.number_labels = 250
        args.language_type = 'java'
        args.eval_data_file = '../dataset/test.txt'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.number_labels
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)

    model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.model_name == 'codebert':
        model = CodeBERT(model, config, tokenizer, args)
    elif args.model_name == 'graphcodebert':
        model = GraphCodeBERT(model, config, tokenizer, args)
    elif args.model_name == 'codet5':
        model = CodeT5(model, config, tokenizer, args)

    checkpoint_prefix = 'original/%s_model.bin' % args.model_name
    output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
    model.load_state_dict(torch.load(output_dir))
    model.to(args.device)

    mlm_tokenizer = RobertaTokenizer.from_pretrained('/workspace/Attack/Repair/microsoft/codebert-base-mlm')
    # generate(args, model, tokenizer, mlm_tokenizer, args.train_data_file, 'train')
    generate(args, model, tokenizer, mlm_tokenizer, args.eval_data_file, 'test')


if __name__ == "__main__":
    main()
