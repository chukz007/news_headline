# News Headline

Generate a news headline using generative models

Dataset is here: https://www.kaggle.com/datasets/divyapatel4/microsoft-pens-personalized-news-headlines

File Structure

    ```
    news_headline/
    ├── archive/                   # Contains the downloaded datasets
    ├── results/                   # Stores generated results and outputs
    ├── utilities                  # Project documentation
        ├── __init__.py                # Marks the utilities directory as a Python package
        ├── eval.py                    # Evaluation script for model performance
        ├── hf_model.py                # Hugging Face model implementation
        ├── load_dataset.py            # Script to load and preprocess datasets
        ├── ollama_model.py            # Ollama model integration
        ├── prompts.py                 # Prompt templates for headline generation and 
        ├── translate.py               # Translation interface for multilingual headline output
    ├── kaggle_pens.ipynb          # Jupyter Notebook exploration of dataset
    ├── main.py                    # Main script to run the project
    ├── README.md                  # Project documentation
    ├── requirements.txt           # List of dependencies
    ├── run.sh                     # Shell script to run the project
    ├── test.py                    # Testing and debugging script
    ```

Install Requirements
Create `python=3.10.6` environment for `unbabel-comet` package to work
`pip install -r requirements.txt`

Download the [Ollama](https://ollama.com/) model and setup the model like this  in your command terminal.
`ollama run <llama3.2>`

Run the code
`./run.sh` for mac/linux `run.sh` for windows
