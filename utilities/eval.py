import json
import nltk
#nltk.download('all')
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
smoother = SmoothingFunction()
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import sacrebleu
from comet import download_model, load_from_checkpoint
import torch
from bleurt_pytorch import BleurtConfig, BleurtForSequenceClassification, BleurtTokenizer
from bert_score import score as bertscore

from pathlib import Path
from typing import Iterable, Dict, Tuple, List
from nltk.translate import meteor_score

def _read_records(path):
    """
    Yield dicts from either a JSON array or JSON-Lines file.
    Silently skip lines that can’t be parsed.
    """
    from pathlib import Path, PurePath
    import json, itertools

    path = Path(path)
    with path.open(encoding="utf-8") as fh:
        first = fh.readline().lstrip()
        fh.seek(0)

        # whole file is an array  [...]
        if first.startswith("["):
            try:
                yield from json.load(fh)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"{path} is not valid JSON: {exc}") from None
            return

        # JSONL or mixed file
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"⚠️  skipped line {lineno}: {exc}")



def load_comet_model_once(model_name="Unbabel/wmt22-comet-da"):
    """
    Loads and returns a COMET model (reference-based or referenceless) only once.
    """
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)
    return model


# def evaluate_headline_performance(json_path):
#     """
#     Evaluate BLEU, METEOR, and ROUGE scores from a JSON file containing
#     predictions and references.

#     Args:
#         json_path (str): Path to the JSON file with keys:
#                          'news_body', 'prediction', 'ground_truth'

#     Returns:
#         List of metric scores per item and overall averages.
#     """
#     with open(json_path, 'r', encoding='utf8') as f:
#         data = json.load(f)

#     bleu_scores = []
#     meteor_scores = []
#     rouge_f1_scores = []

#     scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

#     for item in data:
#         ground_truth = item['ground_truth']
#         prediction = item['prediction']

#         # Tokenization
#         gt_tokens = word_tokenize(ground_truth)
#         pred_tokens = word_tokenize(prediction)

#         # METEOR
#         meteor = nltk.translate.meteor_score.single_meteor_score(gt_tokens, pred_tokens)

#         # ROUGE
#         rouge = scorer.score(ground_truth, prediction)
#         rouge_f = sum([
#             rouge['rouge1'].fmeasure,
#             rouge['rouge2'].fmeasure,
#             rouge['rougeL'].fmeasure
#         ]) / 3

#         # BLEU (scaled to 0–1)
#         #bleu = sacrebleu.raw_corpus_bleu([prediction], [[ground_truth]], .01).score / 100

#         # BLEU (safely, with smoothing)
#         bleu = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoother.method1)
#         bleu_scores.append(bleu)  # scale to 0–1 for consistency


#         # Collect scores
#         bleu_scores.append(bleu)
#         meteor_scores.append(meteor)
#         rouge_f1_scores.append(rouge_f)

#     # Compute averages
#     avg_bleu = sum(bleu_scores) / len(bleu_scores)
#     avg_meteor = sum(meteor_scores) / len(meteor_scores)
#     avg_rouge_f1 = sum(rouge_f1_scores) / len(rouge_f1_scores)

#     return avg_bleu, avg_meteor, avg_rouge_f1

def evaluate_headline_performance(json_path: str) -> Tuple[float, float, float]:
    """
    Compute average SacréBLEU, METEOR, and ROUGE-F1 scores for a JSON-Lines file.

    Each line / element must contain:
        - "prediction"   : system headline (string)
        - "ground_truth" : reference headline (string)

    Returns
    -------
    (avg_bleu, avg_meteor, avg_rouge_f1)   # each already on the 0-1 scale
    """
    records = list(_read_records(json_path))      # ← NEW ①
    if not records:                               # ← NEW ②
        raise RuntimeError("No valid JSON objects found in file.")

    # ---------- metric accumulators -------------------------------------------
    bleu_scores, meteor_scores, rouge_f1_scores = [], [], []
    rouge = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    # ---------- main loop ------------------------------------------------------
    for item in records:                          # use parsed list
        reference  = item["ground_truth"]
        hypothesis = item["prediction"]

        # -------- METEOR --------
        meteor = meteor_score.single_meteor_score(
            word_tokenize(reference), word_tokenize(hypothesis)
        )
        meteor_scores.append(meteor)

        # -------- ROUGE-F1 (avg of 1/2/L) --------
        rs = rouge.score(reference, hypothesis)
        rouge_f1 = (rs["rouge1"].fmeasure + rs["rouge2"].fmeasure + rs["rougeL"].fmeasure) / 3
        rouge_f1_scores.append(rouge_f1)

        # -------- SacréBLEU sentence --------
        bleu = (
            sacrebleu.sentence_bleu(
                hypothesis,
                [reference],
                smooth_method="exp",
                tokenize="intl",
                lowercase=False,
            ).score
            / 100.0
        )
        bleu_scores.append(bleu)

    avg_bleu   = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_rouge  = sum(rouge_f1_scores) / len(rouge_f1_scores)
    return avg_bleu, avg_meteor, avg_rouge


def evaluate_semantic_metrics(json_path):
    """
    Evaluate BERTScore and BLEURT (via bleurt-pytorch) from a JSON file,
    with truncation and OOM protection.
    """

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    preds = [item["prediction"] for item in data]
    refs = [item["ground_truth"] for item in data]

    # BERTScore
    _, _, bert_f1 = bertscore(preds, refs, lang="en", verbose=False)
    avg_bertscore = bert_f1.mean().item()

    # BLEURT (safe, line-by-line)
    config = BleurtConfig.from_pretrained('lucadiliello/BLEURT-20-D12')
    model = BleurtForSequenceClassification.from_pretrained('lucadiliello/BLEURT-20-D12')
    tokenizer = BleurtTokenizer.from_pretrained('lucadiliello/BLEURT-20-D12')
    model.eval()

    bleurt_scores = []
    with torch.no_grad():
        for ref, pred in zip(refs, preds):
            inputs = tokenizer(ref, pred, padding='longest', truncation=True, max_length=512, return_tensors='pt')
            output = model(**inputs).logits.flatten().item()
            bleurt_scores.append(output)

    avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)

    return avg_bertscore, avg_bleurt


def evaluate_with_comet(json_path, comet_model, batch_size=8, gpus=1):
    """
    Evaluates a JSON file using a preloaded COMET model (reference-based).

    Args:
        json_path (str): Path to JSON file with 'news_body', 'prediction', and 'ground_truth'.
        comet_model: Preloaded COMET model instance.
        batch_size (int): Batch size for prediction.
        gpus (int): Use GPU if > 0.

    Returns:
        Tuple[list, float]: list of scores, average score
    """
    with open(json_path, 'r', encoding='utf8') as f:
        raw_data = json.load(f)

    comet_ready_data = [
        {"src": item["news_body"], "mt": item["prediction"], "ref": item["ground_truth"]}
        for item in raw_data
    ]

    scores = comet_model.predict(comet_ready_data, batch_size=batch_size, gpus=gpus)
    return scores[0], scores[1]


def evaluate_with_comet_referenceless(data, comet_model, batch_size=8, gpus=1):
    """
    Evaluates a list of translation pairs (src, mt) using a preloaded referenceless COMET model.

    Args:
        data (list): List of dicts with keys 'src' and 'mt'
        comet_model: Preloaded referenceless COMET model
        batch_size (int): Batch size for inference
        gpus (int): Use GPU if > 0

    Returns:
        Tuple[float, List[float]]: (average_score, list_of_scores)
    """
    scores = comet_model.predict(data, batch_size=batch_size, gpus=gpus)
    return scores[0], scores[1]
