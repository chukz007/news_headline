# News Headline

Generate a news headline using generative models

Dataset is here: https://www.kaggle.com/datasets/divyapatel4/microsoft-pens-personalized-news-headlines

File Structure

```
news_headline/
├── archive/                   # Contains the downloaded datasets
├── results/                   # Stores generated results and outputs
├── eval.py                    # Evaluation script for model performance
├── hf_model.py                # Hugging Face model implementation
├── kaggle_pens.ipynb          # Jupyter Notebook exploration of dataset
├── load_dataset.py            # Script to load and preprocess datasets
├── main.py                    # Main script to run the project
├── model.py                   # Model utility functions
├── ollama_model.py            # Ollama model integration
├── README.md                  # Project documentation
├── requirements.txt           # List of dependencies
├── run.job                    # Job runner file
├── run.sh                     # Shell script to run the project
├── test.py                    # Testing and debugging script
```

Install Requirements
`pip install -r requirements.txt`

Run the Code
`run.sh`
