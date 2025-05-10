import nltk
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import sacrebleu
import json
from comet import download_model, load_from_checkpoint
from bert_score import score as bertscore
from bleurt import score as bleurtscore


def evaluate_headline_performance(json_path):
    """
    Evaluate BLEU, METEOR, and ROUGE scores from a JSON file containing
    predictions and references.

    Args:
        json_path (str): Path to the JSON file with keys:
                         'news_body', 'prediction', 'ground_truth'

    Returns:
        List of metric scores per item and overall averages.
    """
    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    bleu_scores = []
    meteor_scores = []
    rouge_f1_scores = []

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    for item in data:
        ground_truth = item['ground_truth']
        prediction = item['prediction']

        # Tokenization
        gt_tokens = word_tokenize(ground_truth)
        pred_tokens = word_tokenize(prediction)

        # METEOR
        meteor = nltk.translate.meteor_score.single_meteor_score(gt_tokens, pred_tokens)

        # ROUGE
        rouge = scorer.score(ground_truth, prediction)
        rouge_f = sum([
            rouge['rouge1'].fmeasure,
            rouge['rouge2'].fmeasure,
            rouge['rougeL'].fmeasure
        ]) / 3

        # BLEU
        bleu = sacrebleu.raw_corpus_bleu([prediction], [[ground_truth]], .01).score

        # Collect scores
        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        rouge_f1_scores.append(rouge_f)

    # Compute averages
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    avg_meteor = sum(meteor_scores) / len(meteor_scores)
    avg_rouge_f1 = sum(rouge_f1_scores) / len(rouge_f1_scores)

    return avg_bleu, avg_meteor, avg_rouge_f1


def evaluate_with_comet(json_path, model_name="Unbabel/wmt22-comet-da", batch_size=8, gpus=1):
    """
    Evaluates a JSON file using the specified COMET model.

    Args:
        json_path (str): Path to the JSON file containing evaluation data.
                         Must include keys: 'news_body', 'prediction', 'ground_truth'
        model_name (str): COMET model to use from Hugging Face Hub.
        batch_size (int): Batch size for prediction.
        gpus (int): Number of GPUs to use (set to 0 for CPU).

    Returns:
        List of COMET scores (one per example).
    """
    # Download and load the COMET model
    model_path = download_model(model_name)
    model = load_from_checkpoint(model_path)

    # Load and transform the result data
    with open(json_path, 'r', encoding='utf8') as f:
        raw_data = json.load(f)

    comet_ready_data = [
        {"src": item["news_body"], "mt": item["prediction"], "ref": item["ground_truth"]}
        for item in raw_data
    ]

    # Predict COMET scores
    model_output = model.predict(comet_ready_data, batch_size=batch_size, gpus=gpus)

    return model_output[0], model_output[1] # return only scores

def evaluate_semantic_metrics(json_path):
    """
    Evaluate BERTScore and BLEURT from a JSON file.
    """

    with open(json_path, 'r', encoding='utf8') as f:
        data = json.load(f)

    preds = [item["prediction"] for item in data]
    refs = [item["ground_truth"] for item in data]

    # BERTScore
    _, _, bert_f1 = bertscore(preds, refs, lang="en", verbose=False)
    avg_bertscore = bert_f1.mean().item()

    # BLEURT
    bleurt_scorer = bleurtscore.Scorer()
    bleurt_scores = bleurt_scorer.score(references=refs, candidates=preds)
    avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)

    return {
        "average_bertscore_f1": avg_bertscore,
        "average_bleurt": avg_bleurt,
        "bertscore_f1_scores": bert_f1.tolist(),
        "bleurt_scores": bleurt_scores
    }
