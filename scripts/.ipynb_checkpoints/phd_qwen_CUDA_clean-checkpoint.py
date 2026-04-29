import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm

# Используем CUDA-версию PHD
from GPTID.IntrinsicDimCUDA_clean import PH, pairwise_distances

# === Настройки PHD ===
MIN_SUBSAMPLE = 10 # 40
INTERMEDIATE_POINTS = 7

# === Загрузка Qwen-модели ===
def load_roberta_model(model_path="FacebookAI/roberta-base", device: str = "cuda:0"):
    tokenizer = RobertaTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = RobertaModel.from_pretrained(
        model_path,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()
    print("Модель загружена на:", model.device)
    return tokenizer, model


def load_qwen_model(model_path: str, token: str, device: str = "cpu"):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        token=token
    )
    model.to(device)
    model.eval()
    print("Модель загружена на:", model.device)
    return tokenizer, model


# === Препроцессинг текста ===
def preprocess_text(text):
    return text.replace('\n', ' ').replace('  ', ' ').strip()

# === Получение эмбеддингов токенов ===
def get_embeds(text, tokenizer, model, max_length=2048000, returns_tokenized=False, raw_input=False, last_hidden_state=-1):
    device = model.device
    if not raw_input:
        text = preprocess_text(text)
    inputs = tokenizer(
        text,
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)
    if last_hidden_state:
        with torch.no_grad():
            embeddings = model(**inputs, output_hidden_states=True).hidden_states[last_hidden_state][0, :, :].half().cpu().numpy()
#         print("embeddings.shape:" , embeddings.shape)
    else:
        with torch.no_grad():
            outp = model(**inputs)

        embeddings = outp[0][0].float().cpu().numpy()

    if not returns_tokenized:
        return embeddings
    
    tokens = [tokenizer.decode([tok]) for tok in inputs['input_ids'].reshape(-1)]
    return embeddings, tokens

# === Один запуск PHD (через solver) ===
def get_phd_single(text, solver, tokenizer, model, max_length=2048000):
    device = model.device
    inputs = tokenizer(
        preprocess_text(text),
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outp = model(**inputs)

    embeddings = outp[0][0].cpu().numpy()
    mx_points = embeddings.shape[0]
    mn_points = MIN_SUBSAMPLE
    embs = torch.tensor(embeddings).to(device)
    dist_matrix = pairwise_distances(embs).cpu().numpy()
    mx_points = embeddings.shape[0]
    mn_points = MIN_SUBSAMPLE
    step = max(1, ( mx_points - mn_points ) // 10)

    return solver.calculate_ph_dim(
        dist_matrix,
        min_points=mn_points,
        max_points=mx_points - step,
        point_jump=step
    )

# === Среднее по нескольким прогонам PHD ===
def get_phd_single_loop(text, solver, tokenizer, model, n_tries=10, max_length=2048000):
    return np.mean([
        get_phd_single(text, solver, tokenizer, model, max_length)
        for _ in range(n_tries)
    ])

# === Вычисление PHD по DataFrame ===
def get_phd(df, tokenizer, model, key='text', is_list=False, alpha=1.0, n_tries=10, max_length=2048000, use_my_phd_estimation=False): # was True
    dims = []
    import os
    
    try:
        os.remove("phds.txt")
    except:
        print("remove is not here")
    solver = PH(use_my_phd_estimation=use_my_phd_estimation, distance_matrix=True)

    for s in tqdm(df[key]):
        import torch
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        try:
            text = s[0] if is_list else s
            phd_value = get_phd_single_loop(text, solver, tokenizer, model, n_tries=n_tries, max_length=max_length)
            dims.append(phd_value)
            with open("phds.txt", 'a') as fd:
                fd.write(str(phd_value) + "\n")
        except Exception as exc:
            dims.append(-1)
            phd_value = -1.0 
            print(exc)
            with open("phds.txt", 'a') as fd:
                fd.write(str(phd_value) + "\n")

    return np.array(dims).reshape(-1, 1)

# === Если уже есть эмбеддинги ===
def get_raw_phd(points, alpha=1.0):
    points = points.T
    mx_points = points.shape[1]
    mn_points = MIN_SUBSAMPLE
    step = max(1, (mx_points - mn_points) // INTERMEDIATE_POINTS)
    solver = PHD(alpha=alpha, metric='euclidean', n_points=9)

    return solver.fit_transform(points.T, min_points=mn_points, max_points=mx_points - step, point_jump=step)

def get_raw_phd_in_loop(points, alpha=1.0, n_tries=10):
    return [get_raw_phd(points, alpha=alpha) for _ in range(n_tries)]
