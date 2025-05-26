import os
import json

def translate_with_ollama(input_path, model_id, sample_size, output_path):
    """
    Translate the predictions in the input JSON file using the specified model.
    Resumes from previous run and writes incrementally.

    Args:
        input_path (str): Path to the input JSON file.
        model_id (str): Model ID for the translation model.
        sample_size (int): Number of samples to process.
        output_path (str): Directory to save the translation_result.json file.
    Returns:
        list: Previously translated and newly added translations.
    """
    from utilities.ollama_model import OllamaModel
    from utilities.prompts import translation_system_prompt, translation_human_prompt

    # Load full dataset
    with open(input_path, "r", encoding="utf8") as f:
        data = json.load(f)

    # Load already-translated results (if file exists)
    output_file = os.path.join(output_path, "translation_result.json")
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf8") as f:
            already_translated = json.load(f)
    else:
        already_translated = []

    # Create a set of already translated predictions (by prediction_en string)
    done = {item["prediction_en"] for item in already_translated}

    target_languages = ["French", "Spanish", "Portuguese", "Italian", "German", "Russian", "Chinese", "Hindi", "Arabic", "Igbo"]

    # Prepare an unused list of translations (kept for future flexibility)
    # translated = []

    # Skip already processed
    start = len(done)
    print(f"⏳ Resuming from index {start} up to {sample_size}...")

    for item in data[start:sample_size]:
        prediction = item.get("prediction")
        ground_truth = item.get("ground_truth")

        if prediction in done:
            continue

        translations = {}
        for lang in target_languages:
            # Format the system prompt for this language
            system_prompt_lang = translation_system_prompt.format(language=lang)
            model = OllamaModel(model_name=model_id, temperature=0).model_(system_prompt_lang)

            # Build the full prompt and translate
            full_prompt = translation_human_prompt.format(language=lang, headline=prediction)
            output = model.invoke({"input": full_prompt}).content.strip()
            translations[f"prediction_{lang[:2].lower()}"] = output.split("\n")[0].strip()

        # Unused, preserved in case needed for batch-based post-processing
        # translated.append({
        #     "ground_truth": ground_truth,
        #     "prediction_en": prediction,
        #     **translations
        # })

        translated_item = {
            "ground_truth": ground_truth,
            "prediction_en": prediction,
            **translations
        }

        # Append and save incrementally
        already_translated.append(translated_item)
        with open(output_file, "w", encoding="utf8") as f:
            json.dump(already_translated, f, indent=4, ensure_ascii=False)

    return already_translated



def translate_with_hf(input_path, model_id, hf_token, sample_size):
    """
    Translate the predictions in the input JSON file using the specified Hugging Face model.
    Args:
        input_path (str): Path to the input JSON file.
        model_id (str): Model ID for the translation model.
        hf_token (str): Hugging Face authentication token.
        sample_size (int): Number of samples to process. Default is 5.
    Returns:
        list: A list of dictionaries containing the translated predictions.
    """
    from utilities.hf_model import Model, Inferencer
    from utilities.prompts import translation_system_prompt, translation_human_prompt

    with open(input_path, "r", encoding="utf8") as f:
        data = json.load(f)

    model = Model(model_id=model_id, hf_auth=hf_token, max_length=256)
    inferencer = Inferencer(model)

    target_languages = ["French", "Spanish", "Portuguese" , "Italian", "German", "Russian", "Chinese", "Hindi", "Arabic"]
    translated = []

    for item in data[:sample_size]:
        prediction = item.get("prediction")
        ground_truth = item.get("ground_truth")

        # Skip if already translated
        if all(f"prediction_{lang[:2].lower()}" in item for lang in target_languages):
            translated.append(item)
            continue

        translations = {}
        for lang in target_languages:
            full_prompt = translation_system_prompt.format(language=lang) + "\n" + \
                          translation_human_prompt.format(language=lang, headline=prediction)
            output = inferencer.evaluate(full_prompt)
            translations[f"prediction_{lang[:2].lower()}"] = output.strip().split("\n")[0]

        translated.append({
            "ground_truth": ground_truth,
            "prediction_en": prediction,
            **translations
        })

    return translated
