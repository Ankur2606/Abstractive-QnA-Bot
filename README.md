## Project Overview
This project is an Abstractive Question-Answering Model powered by Pinecone for vector storage and retrieval, leveraging the ELI5 BART model for sequence-to-sequence sentence formatting. Additionally, it includes an all-dataset retriever model to encode and decode embeddings. 

## Features
- Abstractive Question-Answering: The model can generate human-like answers to user's questions.
- Pinecone Integration: We use Pinecone to efficiently store and retrieve vectors for improved performance.
- ELI5 BART Model: The model leverages BART for sentence formatting, making answers more comprehensible.
- Dataset Retriever: An all-dataset retriever helps with encoding and decoding embeddings for quick data retrieval.

# How to run?
### STEPS:

Clone the repository

```bash
https://github.com/Ankur2606/Abstractive-QnA-Bot.git
```
### STEP 01- Create a conda environment after opening the repository

```bash
conda create -n qna python=3.8  -y
```

```bash
conda activate qna
```


### STEP 02- install the requirements
```bash
pip install -r requirements.txt
```


```bash
# Finally run the following command
python app.py
```

Now,
```bash
open up you local host and port
```


```bash
Author: Bhavya Pratap Singh Tomar
Student at MITS
Email: bhavyapratapsinghtomar@gmail.com

```