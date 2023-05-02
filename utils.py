import torch
import torch.nn as nn
import random
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import os
import re
import numpy as np
import csv
from python_parser.run_parser import get_example_batch, get_identifiers
from keyword import iskeyword
import pycparser
from sklearn.metrics import roc_curve, auc

python_keywords = ['import', '', '[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">",
                   '+', '-', '*', '/', 'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break',
                   'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global',
                   'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try',
                   'while', 'with', 'yield']
java_keywords = ["abstract", "assert", "boolean", "break", "byte", "case", "catch", "do", "double", "else", "enum",
                 "extends", "final", "finally", "float", "for", "goto", "if", "implements", "import", "instanceof",
                 "int", "interface", "long", "native", "new", "package", "private", "protected", "public", "return",
                 "short", "static", "strictfp", "super", "switch", "throws", "transient", "try", "void", "volatile",
                 "while"]
java_special_ids = ["main", "args", "Math", "System", "Random", "Byte", "Short", "Integer", "Long", "Float", "Double",
                    "Character", "Boolean", "Data", "ParseException", "SimpleDateFormat", "Calendar", "Object",
                    "String", "StringBuffer", "StringBuilder", "DateFormat", "Collection", "List", "Map", "Set",
                    "Queue", "ArrayList", "HashSet", "HashMap"]
c_keywords = ["auto", "break", "case", "char", "const", "continue", "default", "do", "double", "else", "enum", "extern",
              "float", "for", "goto", "if", "inline", "int", "long", "register", "restrict", "return", "short",
              "signed", "sizeof", "static", "struct", "switch", "typedef", "union", "unsigned", "void", "volatile",
              "while", "_Alignas", "_Alignof", "_Atomic", "_Bool", "_Complex", "_Generic", "_Imaginary", "_Noreturn",
              "_Static_assert", "_Thread_local", "__func__"]
c_macros = ["NULL", "_IOFBF", "_IOLBF", "BUFSIZ", "EOF", "FOPEN_MAX", "TMP_MAX", "FILENAME_MAX", "L_tmpnam", "SEEK_CUR",
            "SEEK_END", "SEEK_SET", "NULL", "EXIT_FAILURE", "EXIT_SUCCESS", "RAND_MAX", "MB_CUR_MAX"]
c_special_ids = ["main", "stdio", "cstdio", "stdio.h", "size_t", "FILE", "fpos_t", "stdin", "stdout", "stderr",
                 "remove", "rename", "tmpfile", "tmpnam", "fclose", "fflush", "fopen", "freopen", "setbuf", "setvbuf",
                 "fprintf", "fscanf", "printf", "scanf", "snprintf", "sprintf", "sscanf", "vprintf", "vscanf",
                 "vsnprintf", "vsprintf", "vsscanf", "fgetc", "fgets", "fputc", "getc", "getchar", "putc", "putchar",
                 "puts", "ungetc", "fread", "fwrite", "fgetpos", "fseek", "fsetpos", "ftell", "rewind", "clearerr",
                 "feof", "ferror", "perror", "getline", "stdlib", "cstdlib", "stdlib.h", "size_t", "div_t", "ldiv_t",
                 "lldiv_t", "atof", "atoi", "atol", "atoll", "strtod", "strtof", "strtold", "strtol", "strtoll",
                 "strtoul", "strtoull", "rand", "srand", "aligned_alloc", "calloc", "malloc", "realloc", "free",
                 "abort", "atexit", "exit", "at_quick_exit", "_Exit", "getenv", "quick_exit", "system", "bsearch",
                 "qsort", "abs", "labs", "llabs", "div", "ldiv", "lldiv", "mblen", "mbtowc", "wctomb", "mbstowcs",
                 "wcstombs", "string", "cstring", "string.h", "memcpy", "memmove", "memchr", "memcmp", "memset",
                 "strcat", "strncat", "strchr", "strrchr", "strcmp", "strncmp", "strcoll", "strcpy", "strncpy",
                 "strerror", "strlen", "strspn", "strcspn", "strpbrk", "strstr", "strtok", "strxfrm", "memccpy",
                 "mempcpy", "strcat_s", "strcpy_s", "strdup", "strerror_r", "strlcat", "strlcpy", "strsignal",
                 "strtok_r", "iostream", "istream", "ostream", "fstream", "sstream", "iomanip", "iosfwd", "ios", "wios",
                 "streamoff", "streampos", "wstreampos", "streamsize", "cout", "cerr", "clog", "cin", "boolalpha",
                 "noboolalpha", "skipws", "noskipws", "showbase", "noshowbase", "showpoint", "noshowpoint", "showpos",
                 "noshowpos", "unitbuf", "nounitbuf", "uppercase", "nouppercase", "left", "right", "internal", "dec",
                 "oct", "hex", "fixed", "scientific", "hexfloat", "defaultfloat", "width", "fill", "precision", "endl",
                 "ends", "flush", "ws", "showpoint", "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "sinh",
                 "cosh", "tanh", "exp", "sqrt", "log", "log10", "pow", "powf", "ceil", "floor", "abs", "fabs", "cabs",
                 "frexp", "ldexp", "modf", "fmod", "hypot", "ldexp", "poly", "matherr"]
special_char = ['[', ']', ':', ',', '.', '(', ')', '{', '}', 'not', 'is', '=', "+=", '-=', "<", ">", '+', '-', '*', '/',
                '|']
__ops__ = ["...", ">>=", "<<=", "+=", "-=", "*=", "/=", "%=", "&=", "^=", "|=", ">>", "<<", "++", "--", "->", "&&",
           "||", "<=", ">=", "==", "!=", ";", "{", "<%", "}", "%>", ",", ":", "=", "(", ")", "[", "<:", "]", ":>", ".",
           "&", "!", "~", "-", "+", "*", "/", "%", "<", ">", "^", "|", "?"]
__parser__ = None


def select_parents(population):
    length = range(len(population))
    index_1 = random.choice(length)
    index_2 = random.choice(length)
    chromesome_1 = population[index_1]
    chromesome_2 = population[index_2]
    return chromesome_1, index_1, chromesome_2, index_2


def mutate(chromesome, variable_substitue_dict):
    tgt_index = random.choice(range(len(chromesome)))
    tgt_word = list(chromesome.keys())[tgt_index]
    chromesome[tgt_word] = random.choice(variable_substitue_dict[tgt_word])
    return chromesome


def crossover(csome_1, csome_2, r=None):
    if r is None:
        r = random.choice(range(len(csome_1)))
    child_1 = {}
    child_2 = {}
    for index, variable_name in enumerate(csome_1.keys()):
        if index < r:
            child_2[variable_name] = csome_1[variable_name]
            child_1[variable_name] = csome_2[variable_name]
        else:
            child_1[variable_name] = csome_1[variable_name]
            child_2[variable_name] = csome_2[variable_name]
    return child_1, child_2


def map_chromesome(chromesome: dict, code: str, lang: str) -> str:
    temp_replace = get_example_batch(code, chromesome, lang)
    return temp_replace


def is_valid_variable_python(name: str) -> bool:
    return name.isidentifier() and not iskeyword(name) and (name not in python_keywords)


def is_valid_variable_java(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in java_keywords:
        return False
    elif name in java_special_ids:
        return False
    return True


def is_valid_variable_c(name: str) -> bool:
    if not name.isidentifier():
        return False
    elif name in c_keywords:
        return False
    elif name in c_macros:
        return False
    elif name in c_special_ids:
        return False
    return True


def is_valid_variable_name(name: str, lang: str) -> bool:
    if lang == 'python':
        return is_valid_variable_python(name)
    elif lang == 'c':
        return is_valid_variable_c(name)
    elif lang == 'java':
        return is_valid_variable_java(name)
    else:
        return False


def is_valid_substitue(substitute: str, tgt_word: str, lang: str) -> bool:
    is_valid = True
    if not is_valid_variable_name(substitute, lang):
        is_valid = False
    return is_valid


def _tokenize(seq, tokenizer):
    seq = seq.replace('\n', '')
    words = seq.split(' ')
    sub_words = []
    keys = []
    index = 0
    for word in words:
        sub = tokenizer.tokenize(word)
        sub_words += sub
        keys.append([index, index + len(sub)])
        index += len(sub)
    return words, sub_words, keys


def get_identifier_posistions_from_code(words_list: list, variable_names: list) -> dict:
    positions = {}
    for name in variable_names:
        for index, token in enumerate(words_list):
            if name == token:
                try:
                    positions[name].append(index)
                except:
                    positions[name] = [index]
    return positions


def get_bpe_substitues(substitutes, tokenizer, mlm_model):
    substitutes = substitutes[0:12, 0:4]
    all_substitutes = []
    for i in range(substitutes.size(0)):
        if len(all_substitutes) == 0:
            lev_i = substitutes[i]
            all_substitutes = [[int(c)] for c in lev_i]
        else:
            lev_i = []
            for all_sub in all_substitutes[:24]:
                for j in substitutes[i]:
                    lev_i.append(all_sub + [int(j)])
            all_substitutes = lev_i
    c_loss = nn.CrossEntropyLoss(reduction='none')
    all_substitutes = torch.tensor(all_substitutes)
    all_substitutes = all_substitutes[:24].to('cuda')
    N, L = all_substitutes.size()
    word_predictions = mlm_model(all_substitutes)[0]
    ppl = c_loss(word_predictions.view(N * L, -1), all_substitutes.view(-1))
    ppl = torch.exp(torch.mean(ppl.view(N, L), dim=-1))
    _, word_list = torch.sort(ppl)
    word_list = [all_substitutes[i] for i in word_list]
    final_words = []
    for word in word_list:
        tokens = [tokenizer._convert_id_to_token(int(i)) for i in word]
        text = tokenizer.convert_tokens_to_string(tokens)
        final_words.append(text)
    return final_words


def get_substitues(substitutes, tokenizer, mlm_model, use_bpe, substitutes_score=None, threshold=3.0):
    words = []
    sub_len, k = substitutes.size()
    if sub_len == 0:
        return words
    elif sub_len == 1:
        for (i, j) in zip(substitutes[0], substitutes_score[0]):
            if threshold != 0 and j < threshold:
                break
            words.append(tokenizer._decode([int(i)]))
    else:
        if use_bpe == 1:
            words = get_bpe_substitues(substitutes, tokenizer, mlm_model)
        else:
            return words
    return words


def get_masked_code_by_position(tokens: list, positions: dict):
    masked_token_list = []
    replace_token_positions = []
    for variable_name in positions.keys():
        for pos in positions[variable_name]:
            masked_token_list.append(tokens[0:pos] + ['<unk>'] + tokens[pos + 1:])
            replace_token_positions.append(pos)
    return masked_token_list, replace_token_positions


def build_vocab(codes, limit=5000):
    vocab_cnt = {"<str>": 0, "<char>": 0, "<int>": 0, "<fp>": 0}
    for c in tqdm(codes):
        for t in c:
            if len(t) > 0:
                if t[0] == '"' and t[-1] == '"':
                    vocab_cnt["<str>"] += 1
                elif t[0] == "'" and t[-1] == "'":
                    vocab_cnt["<char>"] += 1
                elif t[0] in "0123456789.":
                    if 'e' in t.lower():
                        vocab_cnt["<fp>"] += 1
                    elif '.' in t:
                        if t == '.':
                            if t not in vocab_cnt.keys():
                                vocab_cnt[t] = 0
                            vocab_cnt[t] += 1
                        else:
                            vocab_cnt["<fp>"] += 1
                    else:
                        vocab_cnt["<int>"] += 1
                elif t in vocab_cnt.keys():
                    vocab_cnt[t] += 1
                else:
                    vocab_cnt[t] = 1
    vocab_cnt = sorted(vocab_cnt.items(), key=lambda x: x[1], reverse=True)
    idx2txt = ["<unk>"] + ["<pad>"] + [it[0] for index, it in enumerate(vocab_cnt) if index < limit - 1]
    txt2idx = {}
    for idx in range(len(idx2txt)):
        txt2idx[idx2txt[idx]] = idx
    return idx2txt, txt2idx


def getTensor(batch, batchfirst=False):
    inputs, labels = batch['x'], batch['y']
    inputs, labels = torch.tensor(inputs, dtype=torch.long).cuda(), \
                     torch.tensor(labels, dtype=torch.long).cuda()
    if batchfirst:
        return inputs, labels
    inputs = inputs.permute([1, 0])
    return inputs, labels


def tokens2seq(_tokens):
    seq = ""
    for t in _tokens:
        if t == "<INT>":
            seq += "0 "
        elif t == "<FP>":
            seq += "0. "
        elif t == "<STR>":
            seq += "\"\" "
        elif t == "<CHAR>":
            seq += "'\0' "
        else:
            while "<__SPACE__>" in t:
                t.replace("<__SPACE__>", " ")
            while "<__BSLASH_N__>" in t:
                t.replace("<__BSLASH_N__>", "\n")
            while "<__BSLASH_R__>" in t:
                t.replace("<__BSLASH_R__>", "\r")
            seq += t + " "
    return seq


def getAST(_seq=""):
    global __parser__
    if __parser__ is None:
        __parser__ = pycparser.CParser()
    _ast = __parser__.parse(_seq)
    return _ast


def getDecl(_seq="", _syms={}):
    _node = getAST(_seq)
    if isinstance(_node, pycparser.c_ast.Decl):
        if isinstance(_node.children()[0][1], pycparser.c_ast.TypeDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.PtrDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.ArrayDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.FuncDecl):
            _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.Struct):
            _syms.add(_node.children()[0][1].name)
            if not _node.name is None:
                _syms.add(_node.name)
        elif isinstance(_node.children()[0][1], pycparser.c_ast.Union):
            _syms.add(_node.children()[0][1].name)
            if not _node.name is None:
                _syms.add(_node.name)
    try:
        for _child in _node.children():
            _syms = getDecl(_child[1], _syms)
    except:
        _node.show()
    return _syms


def isUID(_text=""):
    _text = _text.strip()
    if _text == '':
        return False
    if " " in _text or "\n" in _text or "\r" in _text:
        return False
    elif _text in c_keywords:
        return False
    elif _text in __ops__:
        return False
    elif _text in c_macros:
        return False
    elif _text in c_special_ids:
        return False
    elif _text[0].lower() in "0123456789":
        return False
    elif "'" in _text or '"' in _text:
        return False
    elif _text[0].lower() in "abcdefghijklmnopqrstuvwxyz_":
        for _c in _text[1:-1]:
            if _c.lower() not in "0123456789abcdefghijklmnopqrstuvwxyz_":
                return False
    else:
        return False
    return True


def getUID(_tokens=[], uids=[]):
    ids = {}
    for i, t in enumerate(_tokens):
        if isUID(t) and t in uids[0].keys():
            if t in ids.keys():
                ids[t].append(i)
            else:
                ids[t] = [i]
    return ids


class CodeDataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


class GraphCodeDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args = args

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        attn_mask = np.zeros((self.args.code_length + self.args.data_flow_length,
                              self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        node_index = sum([i > 1 for i in self.examples[item].position_idx])
        max_length = sum([i != 1 for i in self.examples[item].position_idx])
        attn_mask[:node_index, :node_index] = True
        for idx, i in enumerate(self.examples[item].input_ids):
            if i in [0, 2]:
                attn_mask[idx, :max_length] = True
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code):
            if a < node_index and b < node_index:
                attn_mask[idx + node_index, a:b] = True
                attn_mask[a:b, idx + node_index] = True
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx):
                    attn_mask[idx + node_index, a + node_index] = True
        return (torch.tensor(self.examples[item].input_ids),
                torch.tensor(attn_mask),
                torch.tensor(self.examples[item].position_idx),
                torch.tensor(self.examples[item].label))


class CodeT5Dataset(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids), torch.tensor(self.examples[i].label)


class CodePairDataset(Dataset):
    def __init__(self, examples, args):
        self.examples = examples
        self.args = args

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, item):
        attn_mask_1 = np.zeros((self.args.code_length + self.args.data_flow_length,
                                self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        node_index = sum([i > 1 for i in self.examples[item].position_idx_1])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_1])
        attn_mask_1[:node_index, :node_index] = True
        for idx, i in enumerate(self.examples[item].input_ids_1):
            if i in [0, 2]:
                attn_mask_1[idx, :max_length] = True
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_1):
            if a < node_index and b < node_index:
                attn_mask_1[idx + node_index, a:b] = True
                attn_mask_1[a:b, idx + node_index] = True
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_1):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx_1):
                    attn_mask_1[idx + node_index, a + node_index] = True
        attn_mask_2 = np.zeros((self.args.code_length + self.args.data_flow_length,
                                self.args.code_length + self.args.data_flow_length), dtype=np.bool)
        node_index = sum([i > 1 for i in self.examples[item].position_idx_2])
        max_length = sum([i != 1 for i in self.examples[item].position_idx_2])
        attn_mask_2[:node_index, :node_index] = True
        for idx, i in enumerate(self.examples[item].input_ids_2):
            if i in [0, 2]:
                attn_mask_2[idx, :max_length] = True
        for idx, (a, b) in enumerate(self.examples[item].dfg_to_code_2):
            if a < node_index and b < node_index:
                attn_mask_2[idx + node_index, a:b] = True
                attn_mask_2[a:b, idx + node_index] = True
        for idx, nodes in enumerate(self.examples[item].dfg_to_dfg_2):
            for a in nodes:
                if a + node_index < len(self.examples[item].position_idx_2):
                    attn_mask_2[idx + node_index, a + node_index] = True
        return (torch.tensor(self.examples[item].input_ids_1),
                torch.tensor(self.examples[item].position_idx_1),
                torch.tensor(attn_mask_1),
                torch.tensor(self.examples[item].input_ids_2),
                torch.tensor(self.examples[item].position_idx_2),
                torch.tensor(attn_mask_2),
                torch.tensor(self.examples[item].label))


def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class Recorder:
    def __init__(self, file_path: str) -> None:
        self.file_path = file_path
        self.f = open(file_path, 'w')
        self.writer = csv.writer(self.f)
        self.writer.writerow(
            ["Index", "Original Code", "Program Length", "Adversarial Code", "True Label", "Original Prediction",
             "Adv Prediction", "Is Success", "Extracted Names", "Importance Score", "No. Changed Names",
             "No. Changed Tokens", "Replaced Names", "Attack Type", "Query Times", "Time Cost"])

    def write(self, index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success, variable_names,
              score_info, nb_changed_var, nb_changed_pos, replace_info, attack_type, query_times, time_cost):
        self.writer.writerow([index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success,
                              ",".join(variable_names), score_info, nb_changed_var, nb_changed_pos, replace_info,
                              attack_type,
                              query_times, time_cost])

    def writemhm(self, index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success,
                 variable_names, score_info, nb_changed_var, nb_changed_pos, replace_info, attack_type, query_times,
                 time_cost):
        self.writer.writerow([index, code, prog_length, adv_code, true_label, orig_label, temp_label, is_success,
                              variable_names, score_info, nb_changed_var, nb_changed_pos, replace_info, attack_type,
                              query_times, time_cost])


def get_edits_similarity(code1, code2, fasttext_model, language_type):
    variable_names1, function_names1, _ = get_identifiers(code1, language_type)
    variable_names2, function_names2, _ = get_identifiers(code2, language_type)
    sims = [1]
    edits = 0
    for i in range(min(len(variable_names1), len(variable_names2))):
        if variable_names1[i] != variable_names2[i]:
            edits += 1
        else:
            continue
        vec_1 = fasttext_model.get_word_vector(variable_names1[i])
        vec_2 = fasttext_model.get_word_vector(variable_names2[i])
        vec_1 = np.mat(vec_1)
        vec_2 = np.mat(vec_2)
        num = float(vec_1 * vec_2.T)
        denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
        cos = num / denom
        sim = cos
        sims.append(sim)
    for i in range(min(len(function_names1), len(function_names2))):
        if function_names1[i] != function_names2[i]:
            edits += 1
        else:
            continue
        vec_1 = fasttext_model.get_word_vector(function_names1[i])
        vec_2 = fasttext_model.get_word_vector(function_names2[i])
        vec_1 = np.mat(vec_1)
        vec_2 = np.mat(vec_2)
        num = float(vec_1 * vec_2.T)
        denom = np.linalg.norm(vec_1) * np.linalg.norm(vec_2)
        cos = num / denom
        sim = cos
        sims.append(sim)
    return edits, np.average(sims)


def getTokenIndexList2(input_tokens, code_tokens):
    index = 0
    temp = ''
    temp2 = code_tokens[index]
    tokenIndexList = []
    start_index = 1
    for i in range(1, len(input_tokens) - 1):
        if input_tokens[i][0] == 'Ġ':
            temp += input_tokens[i][1:]
        else:
            temp += input_tokens[i]
        if temp == temp2:
            tokenIndexList.append([temp, start_index, i])
            start_index = i + 1
            temp = ''
            index += 1
            if index == len(code_tokens):
                break
            temp2 = code_tokens[index]
        else:
            while len(temp) > len(temp2):
                index += 1
                temp2 += code_tokens[index]
            if temp == temp2:
                tokenIndexList.append([temp, start_index, i])
                start_index = i + 1
                temp = ''
                index += 1
                if index == len(code_tokens):
                    break
                temp2 = code_tokens[index]
    return tokenIndexList


def getTokenIndexList(input_tokens, identifiers):
    temp_input_tokens = []
    for i in input_tokens:
        if len(i) > 1 and i[0] == 'Ġ':
            temp_input_tokens.append(i[1:])
        else:
            temp_input_tokens.append(i)
    tokenIndexList = []
    for iden in identifiers:
        i = 1
        while i < len(temp_input_tokens) - 1:
            temp = temp_input_tokens[i]
            if temp == iden:
                tokenIndexList.append([iden, i, i])
                i += 1
            elif temp == iden[:len(temp)]:
                start_index = i
                Flag = False
                while len(temp) < len(iden) and i < len(temp_input_tokens) - 2:
                    i += 1
                    temp += temp_input_tokens[i]
                    if temp == iden:
                        tokenIndexList.append([iden, start_index, i])
                        Flag = True
                        break
                if not Flag:
                    i = start_index + 1
            else:
                i += 1

    return tokenIndexList


def output_weights(attentions, input_tokens, code_tokens):
    layer_attention_list = []
    tokenIndexList = getTokenIndexList(input_tokens, code_tokens)
    for batch_j in range(0, len(attentions[0])):
        for layer_i in range(len(attentions) - 1, len(attentions)):
            layer_attention_list.append(
                output_layer_attention(layer_i, attentions[layer_i][batch_j].cpu().numpy(), tokenIndexList))
    layer_attention_list = layer_attention_list[0]
    return layer_attention_list


def output_layer_attention(layer_i, attentions, tokenIndexList):
    layer_attention_list = []
    for identifier_range in tokenIndexList:
        identifier = identifier_range[0]
        identifier_start = identifier_range[1]
        identifier_end = identifier_range[2]
        current_attention_list = [0.0] * len(attentions[0][0])
        for head in attentions:
            for i in range(0, len(head)):
                for j in range(identifier_start, identifier_end + 1):
                    current_attention_list[j] += head[i][j]
        layer_attention_list.append(
            [identifier, np.sum(current_attention_list) / len(attentions[0][0]) / len(attentions)])
    return layer_attention_list


def softmax_entropy(data, args):
    scores = []
    labels = []
    for i in data:
        confidence = np.array(i[0])
        y_pred = i[1]
        y_true = i[2]
        labels.append(int(y_pred != y_true))
        increments = np.zeros(shape=confidence.shape, dtype=np.float32)
        indexes_of_zeros = confidence == 0
        increments[indexes_of_zeros] = 1e-20
        nonzero_data = confidence + increments
        temp_scores = [-np.mean(confidence * np.log2(nonzero_data))]
        scores.append(np.average(temp_scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    for i in range(len(thresholds)):
        print(thresholds[i], str(round(tpr[i] * 100, 2)) + '%', str(round(fpr[i] * 100, 2)) + '%')
    print('auc', round(auc_val, 4))


def vanilla_softmax(data, args):
    scores = []
    labels = []
    for i in data:
        confidence = i[0]
        y_pred = i[1]
        y_true = i[2]
        labels.append(int(y_pred != y_true))
        temp_scores = [1 - np.max(confidence)]
        scores.append(np.average(temp_scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    for i in range(len(thresholds)):
        print(thresholds[i], str(round(tpr[i] * 100, 2)) + '%', str(round(fpr[i] * 100, 2)) + '%')
    print('auc', round(auc_val, 4))


def prediction_confidence_score(data, args):
    scores = []
    labels = []
    for i in data:
        confidence = i[0]
        y_pred = i[1]
        y_true = i[2]
        labels.append(int(y_pred != y_true))
        maxx = np.max(confidence)
        confidence[y_pred] = -1
        secondmaxx = np.max(confidence)
        temp_scores = [1 - (maxx - secondmaxx)]
        scores.append(np.average(temp_scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    for i in range(len(thresholds)):
        print(thresholds[i], str(round(tpr[i] * 100, 2)) + '%', str(round(fpr[i] * 100, 2)) + '%')
    print('auc', round(auc_val, 4))


def deepgini(data, args, log=True):
    scores = []
    labels = []
    for i in data:
        confidence = i[0]
        y_pred = i[1]
        y_true = i[2]
        labels.append(int(y_pred != y_true))
        temp_scores = [1 - np.sum(confidence ** 2)]
        scores.append(np.average(temp_scores))
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    if log:
        for i in range(len(thresholds)):
            print(thresholds[i], str(round(tpr[i] * 100, 2)) + '%', str(round(fpr[i] * 100, 2)) + '%')
        print('auc', round(auc_val, 4))
    return [labels, scores]


def to_camel_case(x):
    return re.sub('_([a-zA-Z])', lambda m: (m.group(1).upper()), x)


def to_snake_case(x):
    return re.sub('(?<=[a-z])[A-Z]|(?<!^)[A-Z](?=[a-z])', '_\g<0>', x).lower()
