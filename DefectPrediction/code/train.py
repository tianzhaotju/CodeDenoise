from __future__ import absolute_import, division, print_function
import sys
sys.path.append('../../')
sys.path.append('../../python_parser')
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import (AdamW, get_linear_schedule_with_warmup, RobertaModel,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          T5Config, T5ForConditionalGeneration)
from tqdm import tqdm
from model import CodeBERT, GraphCodeBERT, CodeT5
from run import CodeBertTextDataset, GraphCodeBertTextDataset, CodeT5TextDataset
from sklearn.metrics import recall_score, precision_score, f1_score


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


def train(args, train_dataset, model, tokenizer):
    print('training...')
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=0)
    args.max_steps = args.epochs * len(train_dataloader)
    args.save_steps = len(train_dataloader)
    args.warmup_steps = args.max_steps // 5
    model.to(args.device)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps,
                                                num_training_steps=args.max_steps)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    global_step = 0
    tr_loss = 0.0
    best_precision = 0
    model.zero_grad()
    for idx in range(args.epochs):
        bar = tqdm(train_dataloader, total=len(train_dataloader))
        tr_num = 0
        train_loss = 0
        for step, batch in enumerate(bar):
            if args.model_name == 'codebert':
                inputs = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                model.train()
                loss, logits, _ = model(inputs, labels, False)
            elif args.model_name == 'graphcodebert':
                inputs_ids = batch[0].to(args.device)
                attn_mask = batch[1].to(args.device)
                position_idx = batch[2].to(args.device)
                labels = batch[3].to(args.device)
                model.train()
                loss, logits, _ = model(inputs_ids, attn_mask, position_idx, labels, False)
            elif args.model_name == 'codet5':
                inputs = batch[0].to(args.device)
                labels = batch[1].to(args.device)
                model.train()
                loss, logits, _ = model(inputs, labels, False)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            avg_loss = round(train_loss / tr_num, 5)
            bar.set_description("epoch {} loss {}".format(idx, avg_loss))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                if global_step % args.save_steps == 0:
                    results = evaluate(args, model, tokenizer, eval_when_training=True)
                    if results['eval_precision'] > best_precision:
                        best_precision = results['eval_precision']
                        print("  " + "*" * 20)
                        print("  Best acc:%s", round(best_precision, 4))
                        print("  " + "*" * 20)
                        checkpoint_prefix = 'original'
                        output_dir = os.path.join(args.output_dir, '{}'.format(checkpoint_prefix))
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_dir = os.path.join(output_dir, '{}_{}'.format( args.model_name, 'model.bin'))
                        torch.save(model_to_save.state_dict(), output_dir)
                        print("Saving model checkpoint to %s", output_dir)


def evaluate(args, model, tokenizer, eval_when_training=False):
    print('evaluating...')
    if args.model_name == 'codebert':
        eval_dataset = CodeBertTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'graphcodebert':
        eval_dataset = GraphCodeBertTextDataset(tokenizer, args, args.eval_data_file)
    elif args.model_name == 'codet5':
        eval_dataset = CodeT5TextDataset(tokenizer, args, args.eval_data_file)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.eval_batch_size, num_workers=0)

    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    model.eval()
    logits = []
    y_trues = []
    for batch in tqdm(eval_dataloader):
        if args.model_name == 'codebert':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                _, logit, _ = model(inputs, labels, False)
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())
        elif args.model_name == 'graphcodebert':
            inputs_ids = batch[0].to(args.device)
            attn_mask = batch[1].to(args.device)
            position_idx = batch[2].to(args.device)
            label = batch[3].to(args.device)
            with torch.no_grad():
                _, logit, _ = model(inputs_ids, attn_mask, position_idx, label, False)
                logits.append(logit.cpu().numpy())
                y_trues.append(label.cpu().numpy())
        elif args.model_name == 'codet5':
            inputs = batch[0].to(args.device)
            labels = batch[1].to(args.device)
            with torch.no_grad():
                _, logit, _ = model(inputs, labels, False)
                logits.append(logit.cpu().numpy())
                y_trues.append(labels.cpu().numpy())
    logits = np.concatenate(logits, 0)
    y_trues = np.concatenate(y_trues, 0)
    y_preds = []
    for logit in logits:
        y_preds.append(np.argmax(logit))
    count = 0
    for i in range(len(y_preds)):
        if y_preds[i] == y_trues[i]:
            count += 1

    result = {
        "eval_recall": float(recall_score(y_trues, y_preds, average='macro')),
        "eval_precision": float(precision_score(y_trues, y_preds, average='macro')),
        "eval_f1": float(f1_score(y_trues, y_preds, average='macro')),
    }
    for key in sorted(result.keys()):
        print("  %s = %s", key, str(round(result[key], 4)))
    print('******', count / len(y_preds))
    return result


def main():
    # CUDA_VISIBLE_DEVICES=0 python train.py --model_name=codebert --do_train=0;
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default=None, type=str, required=True, help="model name.")
    parser.add_argument("--do_train", default=0, type=int)

    args = parser.parse_args()
    args.output_dir = './saved_models/'
    args.train_data_file = '../dataset/train.txt'

    args.dataset_name = os.path.abspath('.').split('/')[-2].strip()
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
    args.block_size = 512 
    args.seed = 123456
    args.evaluate_during_training = True
    args.train_batch_size = 16
    args.eval_batch_size = 1
    args.max_grad_norm = 1.0
    args.warmup_steps = 0
    args.max_steps = -1
    args.adam_epsilon = 1e-8
    args.weight_decay = 0.0
    args.gradient_accumulation_steps = 1
    if args.model_name == 'codebert':
        args.model_type = 'codebert_roberta'
        args.config_name = 'microsoft/codebert-base'
        args.model_name_or_path = 'microsoft/codebert-base'
        args.tokenizer_name = 'roberta-base'
        args.epochs = 5
        args.learning_rate = 2e-5
    elif args.model_name == 'graphcodebert':
        args.model_type = 'graphcodebert_roberta'
        args.config_name = 'microsoft/graphcodebert-base'
        args.model_name_or_path = 'microsoft/graphcodebert-base'
        args.tokenizer_name = 'microsoft/graphcodebert-base'
        args.epochs = 5
        args.code_length = 448
        args.data_flow_length = 64
        args.learning_rate = 2e-5
    elif args.model_name == 'codet5':
        args.model_type = 'codet5'
        args.config_name = 'Salesforce/codet5-base-multi-sum'
        args.model_name_or_path = 'Salesforce/codet5-base-multi-sum'
        args.tokenizer_name = 'Salesforce/codet5-base-multi-sum'
        args.epochs = 5
        args.learning_rate = 2e-5

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

    if args.do_train:
        if args.model_name == 'codebert':
            train_dataset = CodeBertTextDataset(tokenizer, args, args.train_data_file)
        elif args.model_name == 'graphcodebert':
            train_dataset = GraphCodeBertTextDataset(tokenizer, args, args.train_data_file)
        elif args.model_name == 'codet5':
            train_dataset = CodeT5TextDataset(tokenizer, args, args.train_data_file)
        print('training...')
        train(args, train_dataset, model, tokenizer)
    else:
        result = evaluate(args, model, model, tokenizer)
        for key in sorted(result.keys()):
            print("  %s = %s", key, str(round(result[key], 4)))


if __name__ == "__main__":
    main()