import nltk
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
import sacrebleu



def evaluate_headline_performance(df, index, generated_headline):
    # Extract ground truth headline from DataFrame
    ground_truth = df["Headline"][index]

    # Tokenize the ground truth and generated headlines
    ground_truth_tokens = word_tokenize(ground_truth)
    generated_headline_tokens = word_tokenize(generated_headline)

    # METEOR Score calculation with tokenized inputs
    meteor_score = nltk.translate.meteor_score.single_meteor_score(
        reference=ground_truth_tokens,
        hypothesis=generated_headline_tokens
    )

    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(target=ground_truth, prediction=generated_headline)
    # Calculate the average F-measure
    f_measures = [
        rouge_scores['rouge1'].fmeasure,
        rouge_scores['rouge2'].fmeasure,
        rouge_scores['rougeL'].fmeasure
    ]
    average_f_measure = sum(f_measures) / len(f_measures)

    # BLEU
    bleu_score = sacrebleu.raw_corpus_bleu([generated_headline], [[ground_truth]], .01).score

    # Return all scores
    return bleu_score, meteor_score, rouge_scores, average_f_measure
