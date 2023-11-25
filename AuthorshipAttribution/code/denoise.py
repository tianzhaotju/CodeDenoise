from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../../')
sys.path.append('../../python_parser')
import warnings
warnings.filterwarnings("ignore")
import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from run import MLMTextDataset
from torch.utils.data import DataLoader
from tree_sitter import Language, Parser
from model import CodeBERT, GraphCodeBERT, CodeT5
from parser_folder import DFG_python, DFG_java, DFG_c
from run_parser import get_identifiers_ori, get_example
from utils import CodeDataset, GraphCodeDataset, CodeT5Dataset, is_valid_variable_name, output_weights, set_seed
from transformers import (RobertaModel, RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer, T5Config,
                          T5ForConditionalGeneration, RobertaForMaskedLM, pipeline)
from run import CodeBertInputFeatures, GraphCodeBertInputFeatures, CodeT5InputFeatures, extract_dataflow, \
    codebert_convert_examples_to_features, graphcodebert_convert_examples_to_features, codet5_convert_examples_to_features


dfg_function = {'python': DFG_python, 'java': DFG_java, 'c': DFG_c}
parsers = {}
for lang in dfg_function:
    LANGUAGE = Language('../../python_parser/parser_folder/my-languages.so', lang)
    parser = Parser()
    parser.set_language(LANGUAGE)
    parser = [parser, dfg_function[lang]]
    parsers[lang] = parser
MODEL_CLASSES = {
    'codebert_roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'graphcodebert_roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'codet5': (T5Config, T5ForConditionalGeneration, RobertaTokenizer),
}


def codebert_convert_code_to_features(code, tokenizer, label, args):
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return CodeBertInputFeatures(source_tokens, source_ids, 0, label)


def graphcodebert_convert_code_to_features(code, tokenizer, label, args):
    parser = parsers[args.language_type]
    code_tokens, dfg = extract_dataflow(code, parser, args.language_type)
    code_tokens = [tokenizer.tokenize('@ ' + x)[1:] if idx != 0 else tokenizer.tokenize(x) for idx, x in enumerate(code_tokens)]
    ori2cur_pos = {}
    ori2cur_pos[-1] = (0, 0)
    for i in range(len(code_tokens)):
        ori2cur_pos[i] = (ori2cur_pos[i - 1][1], ori2cur_pos[i - 1][1] + len(code_tokens[i]))
    code_tokens = [y for x in code_tokens for y in x]

    code_tokens = code_tokens[:args.code_length + args.data_flow_length - 2 - min(len(dfg), args.data_flow_length)]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    position_idx = [i + tokenizer.pad_token_id + 1 for i in range(len(source_tokens))]
    dfg = dfg[:args.code_length + args.data_flow_length - len(source_tokens)]
    source_tokens += [x[0] for x in dfg]
    position_idx += [0 for x in dfg]
    source_ids += [tokenizer.unk_token_id for x in dfg]
    padding_length = args.code_length + args.data_flow_length - len(source_ids)
    position_idx += [tokenizer.pad_token_id] * padding_length
    source_ids += [tokenizer.pad_token_id] * padding_length

    reverse_index = {}
    for idx, x in enumerate(dfg):
        reverse_index[x[1]] = idx
    for idx, x in enumerate(dfg):
        dfg[idx] = x[:-1] + ([reverse_index[i] for i in x[-1] if i in reverse_index],)
    dfg_to_dfg = [x[-1] for x in dfg]
    dfg_to_code = [ori2cur_pos[x[1]] for x in dfg]
    length = len([tokenizer.cls_token])
    dfg_to_code = [(x[0] + length, x[1] + length) for x in dfg_to_code]

    return GraphCodeBertInputFeatures(source_tokens, source_ids, position_idx, dfg_to_code, dfg_to_dfg, label)


def codet5_convert_code_to_features(code, tokenizer, label, args):
    code = ' '.join(code.split())
    code_tokens = tokenizer.tokenize(code)[:args.block_size-2]
    source_tokens = [tokenizer.cls_token] + code_tokens + [tokenizer.sep_token]
    source_ids = tokenizer.convert_tokens_to_ids(source_tokens)
    padding_length = args.block_size - len(source_ids)
    source_ids += [tokenizer.pad_token_id] * padding_length
    return CodeT5InputFeatures(source_tokens, source_ids, 0, label)


def is_incorrect(args, original_model, original_tokenizer, all_vars, identifiers, ori_codes, ori_labels, pred_labels):
    results = [pred_labels]
    used_identifiers = set(identifiers)
    replace_examples = []
    for i in range(args.N*2):
        for iden in identifiers:
            gen_idens = []
            count = 0
            random_index = np.random.choice(range(len(all_vars)), len(all_vars), replace=False)
            while len(gen_idens) < args.theta*2 and count < len(random_index):
                temp_gen_iden = all_vars[random_index[count]]
                count += 1
                if is_valid_variable_name(temp_gen_iden, args.language_type) and temp_gen_iden not in gen_idens and temp_gen_iden not in used_identifiers and len(temp_gen_iden) > len(iden):
                    gen_idens.append(temp_gen_iden)
                    used_identifiers.add(temp_gen_iden)
            replace_idens = []
            for gen_iden in gen_idens:
                temp_code = get_example(ori_codes, iden, gen_iden, args.language_type)
                replace_idens.append(gen_iden)
                if args.model_name == 'codebert':
                    new_feature = codebert_convert_examples_to_features(temp_code, ori_labels, original_tokenizer, args)
                elif args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_examples_to_features(temp_code, ori_labels, original_tokenizer, args)
                elif args.model_name == 'codet5':
                    new_feature = codet5_convert_examples_to_features(temp_code, ori_labels, original_tokenizer, args)
                replace_examples.append(new_feature)
    if args.model_name == 'codebert':
        new_dataset = CodeDataset(replace_examples)
    elif args.model_name == 'graphcodebert':
        new_dataset = GraphCodeDataset(replace_examples, args)
    elif args.model_name == 'codet5':
        new_dataset = CodeT5Dataset(replace_examples)
    if len(replace_examples) > 0:
        all_logits, preds, _ = original_model.get_results(new_dataset, args.eval_batch_size, False)
        results.extend(preds)
    temp_y_preds = {}
    for la in results:
        if la in temp_y_preds.keys():
            temp_y_preds[la] += 1
        else:
            temp_y_preds[la] = 1
    rs_pred_labels = sorted(temp_y_preds.items(), key=lambda x: x[1], reverse=True)
    if pred_labels != rs_pred_labels[0] and len(rs_pred_labels) > 1:
        return 1
    return 0


def evaluate(args, mlm_model, mlm_tokenizer, original_model, original_tokenizer, eval_when_training=False):
    print('evaluating...')
    eval_dataset = MLMTextDataset(mlm_tokenizer, args, args.eval_data_file)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, num_workers=0)
    all_vars = [i.strip() for i in open('../dataset/all_vars.txt', 'r').readlines()]
    voc = np.load('../dataset/voc.npy', allow_pickle=True).item()
    mlm_model.eval()
    fill_mask = pipeline('fill-mask', model=mlm_model, tokenizer=mlm_tokenizer, device=0, targets=voc.keys())
    y_preds = []
    y_trues = []
    for batch_i, batch in enumerate(tqdm(eval_dataloader)):
        ori_codes = batch['ori_codes'][0].replace("\\n", "\n").replace("\\t", "\t").replace('\"', '"')
        ori_labels = batch['ori_labels'].cpu().numpy()[0]
        pred_labels = batch['pred_labels'].cpu().numpy()[0]
        mask_codes = batch['mask_codes'][0].replace("\\n", "\n").replace("\\t", "\t").replace('\"', '"')
        t = mlm_tokenizer.tokenize(mask_codes)[:args.block_size - 2]
        mask_codes = mlm_tokenizer.convert_tokens_to_string(t)
        identifiers, _ = get_identifiers_ori(mask_codes, args.language_type)
        if is_incorrect(args, original_model, original_tokenizer, all_vars, identifiers, ori_codes, ori_labels,
                        pred_labels) == 0:
            continue
        y_trues.append(ori_labels)
        weight_path = '../weights/%s/%s.npy' % (args.model_name, str(batch_i))
        if os.path.exists(weight_path):
            temp_identifiers = np.load(weight_path)
        else:
            if args.model_name == 'codebert':
                new_feature = codebert_convert_code_to_features(mask_codes, original_tokenizer, ori_labels, args)
            elif args.model_name == 'graphcodebert':
                new_feature = graphcodebert_convert_code_to_features(mask_codes, original_tokenizer, ori_labels, args)
            elif args.model_name == 'codet5':
                new_feature = codet5_convert_code_to_features(mask_codes, original_tokenizer, ori_labels, args)
            if args.model_name == 'codebert':
                new_dataset = CodeDataset([new_feature])
            elif args.model_name == 'graphcodebert':
                new_dataset = GraphCodeDataset([new_feature], args)
            elif args.model_name == 'codet5':
                new_dataset = CodeT5Dataset([new_feature])
            _, _, attentions = original_model.get_results(new_dataset, args.eval_batch_size)
            attentions = attentions[0]
            attention_weights = output_weights(attentions, new_feature.input_tokens, identifiers)
            temp_identifiers = {}
            for i in identifiers:
                temp_identifiers[i] = 0
                for j in attention_weights:
                    if j[0] == i:
                        temp_identifiers[i] = max(j[1], temp_identifiers[i])
            temp_identifiers = sorted(temp_identifiers.items(), key=lambda x: x[1], reverse=True)
            np.save(weight_path, np.array(temp_identifiers))
        temp_identifiers = [i[0] for i in temp_identifiers]
        assert set(temp_identifiers) == set(identifiers)
        identifiers = temp_identifiers

        min_prob = 1
        used_identifiers = set(identifiers)
        for iden in identifiers:
            used_identifiers.remove(iden)
            temp_mask_codes = get_example(mask_codes, iden, '<mask>', args.language_type)
            t = mlm_tokenizer.tokenize(temp_mask_codes)[:args.block_size - 2]
            temp_mask_codes = mlm_tokenizer.convert_tokens_to_string(t)
            try:
                outputs = fill_mask(temp_mask_codes, top_k=1)
            except:
                continue
            gen_idens = set()
            for i in range(np.shape(outputs)[0]):
                if len(np.shape(outputs)) == 2:
                    for j in range(np.shape(outputs)[1]):
                        temp_gen_iden = outputs[i][j]['token_str'].strip()
                        if temp_gen_iden in voc.keys():
                            temp_gen_iden = voc[temp_gen_iden]
                        if is_valid_variable_name(temp_gen_iden, args.language_type) and temp_gen_iden not in used_identifiers:
                            gen_idens.add(temp_gen_iden)
                elif len(np.shape(outputs)) == 1:
                    temp_gen_iden = outputs[i]['token_str'].strip()
                    if temp_gen_iden in voc.keys():
                        temp_gen_iden = voc[temp_gen_iden]
                    if is_valid_variable_name(temp_gen_iden, args.language_type) and temp_gen_iden not in used_identifiers:
                        gen_idens.add(temp_gen_iden)
            replace_examples = []
            replace_idens = []

            for gen_iden in gen_idens:
                temp_code = get_example(ori_codes, iden, gen_iden, args.language_type)
                replace_idens.append(gen_iden)
                if args.model_name == 'codebert':
                    new_feature = codebert_convert_code_to_features(temp_code, original_tokenizer, ori_labels, args)
                elif args.model_name == 'graphcodebert':
                    new_feature = graphcodebert_convert_code_to_features(temp_code, original_tokenizer, ori_labels, args)
                elif args.model_name == 'codet5':
                    new_feature = codet5_convert_code_to_features(temp_code, original_tokenizer, ori_labels, args)
                replace_examples.append(new_feature)
            if args.model_name == 'codebert':
                new_dataset = CodeDataset(replace_examples)
            elif args.model_name == 'graphcodebert':
                new_dataset = GraphCodeDataset(replace_examples, args)
            elif args.model_name == 'codet5':
                new_dataset = CodeT5Dataset(replace_examples)
            if len(new_dataset) == 0:
                continue
            all_logits, preds, _ = original_model.get_results(new_dataset, args.eval_batch_size, False)
            index = -1
            for i in range(len(preds)):
                if preds[i] != pred_labels:
                    y_preds.append(preds[i])
                    break
                elif min_prob > np.max(all_logits[i]):
                    min_prob = np.max(all_logits[i])
                    index = i
            if index > -1:
                used_identifiers.add(replace_idens[index])
                mask_codes = get_example(mask_codes, iden, replace_idens[index], args.language_type)
                ori_codes = get_example(ori_codes, iden, replace_idens[index], args.language_type)
            if len(y_preds) == len(y_trues):
                break
        if len(y_preds) < len(y_trues):
            y_preds.append(pred_labels)

    results = {'repair': 0, 'incorrect': len(y_preds), 'acc': 0}
    for i in range(len(y_preds)):
        if y_preds[i] == y_trues[i]:
            results['repair'] += 1
    results['acc'] = results['repair']/results['incorrect']
    print('  Repair:', results['repair'])
    print('  Incorrect:', results['incorrect'])
    print('  Acc:', str(round(results['acc'] * 100, 2)) + '%')
    return results


def main():
    # CUDA_VISIBLE_DEVICES=0 python denoise.py --model_name=codebert --theta=1 --N=1;
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=True, help="model name.")
    parser.add_argument("--theta", default=1, type=int)
    parser.add_argument("--N", default=1, type=int)
    args = parser.parse_args()
    args.dataset_name = os.path.abspath('.').split('/')[-2].strip()
    args.train_data_file = '../dataset/mlm_%s_train.txt' % (args.model_name)
    args.eval_data_file = '../dataset/mlm_%s_test.txt' % (args.model_name)
    args.block_size = 512
    args.seed = 123456
    args.evaluate_during_training = True
    if args.dataset_name == 'AuthorshipAttribution':
        args.number_labels = 66
        args.language_type = 'python'
    elif args.dataset_name == 'DefectPrediction':
        args.number_labels = 4
        args.language_type = 'c'
    elif args.dataset_name == 'FunctionalityClassification':
        args.number_labels = 104
        args.language_type = 'c'
    elif args.dataset_name == 'Cplusplus1000':
        args.number_labels = 1000
        args.language_type = 'c'
    elif args.dataset_name == 'Python800':
        args.number_labels = 800
        args.language_type = 'python'
    elif args.dataset_name == 'Java250':
        args.number_labels = 250
        args.language_type = 'java'
    args.output_dir = 'saved_models'
    args.train_batch_size = 4
    args.eval_batch_size = 1
    args.max_grad_norm = 1.0
    args.warmup_steps = 0
    args.max_steps = -1
    args.adam_epsilon = 1e-8
    args.weight_decay = 0.0
    args.gradient_accumulation_steps = 1
    args.learning_rate = 2e-5
    args.epochs = 200
    if args.model_name == 'codebert':
        args.model_type = 'codebert_roberta'
        args.config_name = 'microsoft/codebert-base'
        args.model_name_or_path = 'microsoft/codebert-base'
        args.tokenizer_name = 'roberta-base'
    elif args.model_name == 'graphcodebert':
        args.model_type = 'graphcodebert_roberta'
        args.config_name = 'microsoft/graphcodebert-base'
        args.model_name_or_path = 'microsoft/graphcodebert-base'
        args.tokenizer_name = 'microsoft/graphcodebert-base'
        args.code_length = 448
        args.data_flow_length = 64
    elif args.model_name == 'codet5':
        args.model_type = 'codet5'
        args.config_name = 'Salesforce/codet5-base-multi-sum'
        args.model_name_or_path = 'Salesforce/codet5-base-multi-sum'
        args.tokenizer_name = 'Salesforce/codet5-base-multi-sum'
    args.n_gpu = torch.cuda.device_count()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args)
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path)
    config.num_labels = args.number_labels
    original_tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name)
    original_model = model_class.from_pretrained(args.model_name_or_path, config=config)
    if args.model_name == 'codebert':
        original_model = CodeBERT(original_model, config, original_tokenizer, args)
    elif args.model_name == 'graphcodebert':
        original_model = GraphCodeBERT(original_model, config, original_tokenizer, args)
    elif args.model_name == 'codet5':
        original_model = CodeT5(original_model, config, original_tokenizer, args)
    original_model.load_state_dict(torch.load('./%s/original/%s_model.bin' % (args.output_dir, args.model_name)))
    original_model.to(args.device)

    mlm_tokenizer = RobertaTokenizer.from_pretrained('microsoft/codebert-base-mlm')
    mlm_model = RobertaForMaskedLM.from_pretrained('microsoft/codebert-base-mlm')
    if os.path.exists('./%s/mlm/%s_model.bin' % (args.output_dir, args.model_name)):
        print('Load Model...')
        mlm_model.load_state_dict(torch.load('./%s/mlm/%s_model.bin' % (args.output_dir, args.model_name)))

    mlm_model.to(args.device)
    results = evaluate(args, mlm_model, mlm_tokenizer, original_model, original_tokenizer)


if __name__ == "__main__":
    main()