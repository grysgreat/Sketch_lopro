import torch
from datasets import load_dataset
import numpy as np

# os.makedirs(quantized_model_dir, exist_ok=True)
def get_wikitext2(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    from modelscope import MsDataset


    # traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    traindata =  MsDataset.load('wikitext', subset_name='wikitext-2-raw-v1', split='train')
    testdata =  MsDataset.load('wikitext', subset_name='wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        attention_mask = torch.ones_like(inp)
        traindataset.append({"input_ids": inp, "attention_mask": attention_mask})
    return traindataset, testenc, tokenizer




def get_wikitext2_(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    from modelscope import MsDataset


    # traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    traindata =  MsDataset.load('wikitext', subset_name='wikitext-2-raw-v1', split='train')
    testdata =  MsDataset.load('wikitext', subset_name='wikitext-2-raw-v1', split='test')

    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")
    testenc = tokenizer("\n\n".join(testdata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        traindataset.append((inp, tar))
    return traindataset, testenc, tokenizer



def get_redPajamas(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    from modelscope import MsDataset


    # traindata = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # testdata = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")

    #traindata = MsDataset.load('swift/RedPajama-Data-1T', config_name='wikipedia',split='train')  # 或其他你需要的子集
    #traindata = load_dataset('togethercomputer/RedPajama-Data-1T-Sample', split='train', trust_remote_code=True)
    traindata =load_dataset("mit-han-lab/pile-val-backup", split="validation")
    from transformers import AutoTokenizer

    try:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model, use_fast=True)
    trainenc = tokenizer("\n\n".join(traindata["text"]), return_tensors="pt")

    import random

    random.seed(seed)
    np.random.seed(0)
    torch.random.manual_seed(0)
    
    traindataset = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        traindataset.append((inp, tar))
    print("finish load red pajamas!")
    return traindataset


def get_c4_(nsamples, seed, seqlen, model):
    from datasets import load_dataset

    valdata =  load_dataset("C4", data_files={"validation": "/data/c4-validation.00000-of-00008.json"},split="validation")

    from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained(model, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(model)


    import random
    random.seed(0)
    valenc = []
    for _ in range(256):
        while True:
            i = random.randint(0, len(valdata) - 1)
            tmp = tokenizer(valdata[i]['text'], return_tensors='pt')
            # if tmp.input_ids.shape[1] >= seqlen:
            if tmp.input_ids.shape[1] >= seqlen + 1:
                break
        i = random.randint(0, tmp.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        valenc.append(tmp.input_ids[:, i:j])
    valenc = torch.hstack(valenc)

    class TokenizerWrapper:

        def __init__(self, input_ids):
            self.input_ids = input_ids

    valenc = TokenizerWrapper(valenc)

    return  valenc

def get_pile(data="pileval", tokenizer=None, n_samples=512, block_size=512):
    from modelscope import MsDataset
    if data == "pileval":
        dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
    elif data == "wikitext2":
        #dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        dataset = MsDataset.load('wikitext', subset_name='wikitext-2-raw-v1', split='train')
    elif data == "c4":
        dataset = load_dataset("C4", data_files={"validation": "/data/c4-validation.00000-of-00008.json"},split="validation")
    else:
        raise NotImplementedError
    #dataset = dataset.shuffle(seed=42)
    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data["text"]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > block_size:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    print(f" * Split into {n_split} blocks")
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]
