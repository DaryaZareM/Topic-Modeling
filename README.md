# Topic Modeling Comparison on Persian Economics Tweets Dataset

Compare LLMs, Contextualized models and basic models(LDA,...) in Topic modeling task.

This repository contains code for comparing different topic modeling methods on a dataset of Persian tweets related to economics. The models are evaluated based on various metrics including purity, normalized mutual information (NMI), accuracy, and F1-score.

## Models and Metrics Comparison

The following table summarizes the performance of different topic modeling methods on the dataset:

| Model\Metric            | Purity | NMI   | Accuracy | F1-score |
|-------------------------|--------|-------|----------|----------|
| LDA                     | 84.61  | 67.32 | 81       | 77       |
| STM                     | 84.61  | 62.65 | 79       | 65       |
| CTM (ALBERT)            | 82.41  | 56.57 | 82       | 78       |
| CTM (ParsBERT)          | 80.21  | 51.82 | 80       | 76       |
| ETM                     | 85.71  | 70.91 | 86       | 82       |
| BERTopic (default)      | 83.51  | 62.67 | 84       | 79       |
| ChatGPT (translate-prompt) | 94.5 | 81.06 | 95       | 91       |
| ChatGPT (google-translate) | 95.6 | 85.03 | 96       | 94       |
| ChatGPT                 | 96.7   | 89.86 | 97       | 95       |
| HuggingChat (translate-prompt) | 79.12 | 49.04 | 79 | 67    |
| HuggingChat (google-translate) | 92.13 | 75.39 | 92 | 90    |

## Evaluation Parameters and Models

- **Purity**: Purity measures the extent to which clusters contain only instances of a single class. Higher purity indicates better separation of topics.
- **Normalized Mutual Information (NMI)**: NMI measures the similarity between the true class labels and the predicted clusters, adjusted for chance. Higher NMI values indicate better agreement between true and predicted labels.
- **Accuracy**: Accuracy measures the proportion of correctly predicted instances. In this context, it refers to the accuracy of topic assignments.
- **F1-score**: F1-score is the harmonic mean of precision and recall, providing a balanced measure of model performance.

The models evaluated in this project include Latent Dirichlet Allocation (LDA), Structural Topic Model (STM), Contextualized Topic Model (CTM) with ALBERT and ParsBERT embeddings, Embedding Topic Model (ETM), BERTopic with default settings, and various versions of ChatGPT and HuggingChat models using different prompts.

## Code Files

- `code/01-Choose-Topics(LDA).ipynb`: Jupyter notebook for implementing the LDA topic modeling method and label the data.
- `code/02.1-Base_Models.ipynb`: Jupyter notebook for implementing base models for topic modeling.
- `code/02.2-CTM_NMF_ProdLDA_models.ipynb`: Jupyter notebook for implementing CTM, NMF, and ProdLDA models.
- `code/02.3-ChatGPT-models.ipynb`: Jupyter notebook for implementing ChatGPT models.
- `code/02.3-HuggingChat-models.ipynb`: Jupyter notebook for implementing HuggingChat models.

## Usage

To replicate the experiments and results, follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the `code` directory.
3. Open and run the desired Jupyter notebook(s) in a Python environment with the required dependencies installed.

## Dependencies

The code relies on the following dependencies:

- Jupyter Notebook
- Python 3.x
- Libraries such as scikit-learn, gensim, pytorch, transformers, and others as specified in the notebooks.

## References

In this project, the following articles and resources were utilized:

- [Re-visiting Automated Topic Model Evaluation with Large Language Models](https://arxiv.org/abs/2305.12152): This article provided insights into automated topic model evaluation techniques using large language models.
- [Pre-training is a Hot Topic: Contextualized Document Embeddings Improve Topic Coherence](https://aclanthology.org/2021.acl-short.96/): This article contributed to the understanding and implementation of pre-training techniques with contextualized document embeddings to improve topic coherence.
- [stm: An R Package for Structural Topic Models](https://www.jstatsoft.org/article/view/v091i02): This resource was instrumental in implementing Structural Topic Models (STM) for topic modeling.

These references served as valuable sources of information and guidance throughout the project.


## Acknowledgments

Special thanks to the developers and contributors of the implemented models and libraries used in this project.

## Contact

For any inquiries or feedback, please contact the project maintainer:

Darya Zare
Email: daryaz2079@gmail.com

