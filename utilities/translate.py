import json

def translate_with_ollama(input_path, model_id, sample_size):
    """
    Translate the predictions in the input JSON file using the specified model.
    Args:
        input_path (str): Path to the input JSON file.
        model_id (str): Model ID for the translation model.
        sample_size (int): Number of samples to process. Default is 5.
    Returns:
        list: A list of dictionaries containing the translated predictions.
    """
    from utilities.ollama_model import OllamaModel
    from utilities.prompts import translation_system_prompt, translation_human_prompt

    with open(input_path, "r", encoding="utf8") as f:
        data = json.load(f)

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
            # Format the system prompt for this language
            system_prompt_lang = translation_system_prompt.format(language=lang)
            model = OllamaModel(model_name=model_id, temperature=0).model_(system_prompt_lang)

            # Build the full prompt and translate
            full_prompt = translation_human_prompt.format(language=lang, headline=prediction)
            output = model.invoke({"input": full_prompt}).content.strip()
            translations[f"prediction_{lang[:2].lower()}"] = output.split("\n")[0].strip()

        translated.append({
            "ground_truth": ground_truth,
            "prediction_en": prediction,
            **translations
        })

    return translated


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

