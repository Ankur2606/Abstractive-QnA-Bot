from transformers import BartTokenizer, BartForConditionalGeneration
import torch
from sentence_transformers import SentenceTransformer
from data_preprocessing import data_preprocessing
import pandas as pd
import pinecone
from tqdm.auto import tqdm
from logging import logger
from pprint import pprint
from datasets import load_dataset

# load the dataset
wiki_data = load_dataset(
    'vblagoje/wikipedia_snippets_streamed',
    split='train',
    streaming=True
).shuffle(seed=960)

docs = data_preprocessing(wiki_data).docs
# create a pandas dataframe with the documents we extracted
df = pd.DataFrame(docs)

# set device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
# load the retriever model from huggingface model hub
retriever = SentenceTransformer("flax-sentence-embeddings/all_datasets_v3_mpnet-base", device=device)

# connect to pinecone environment
pinecone.init(
    api_key="YOUR_PINECONE_API_KEY",
    environment="YOUR_PINECONE_ENVIRONMENT" 
)

#Embedding and indexing the documents
index_name = "abstractive-question-answering"

# check if the abstractive-question-answering index exists
if index_name not in pinecone.list_indexes():
    # create the index if it does not exist
    pinecone.create_index(
        index_name,
        dimension= retriever.get_sentence_embedding_dimension(),
        metric="cosine"
    )

# connect to abstractive-question-answering index we created
index = pinecone.Index(index_name)

batch_size = 64

for i in tqdm(range(0, len(df), batch_size)):
    # find end of batch
    i_end = min(i+batch_size, len(df))
    # extract batch
    batch = df.iloc[i:i_end]
    # generate embeddings for batch
    emb = retriever.encode(batch["passage_text"].tolist()).tolist()
    # get metadata
    meta = batch.to_dict(orient="records")
    # create unique IDs
    ids = [f"{idx}" for idx in range(i, i_end)]
    # add all to upsert list
    to_upsert = list(zip(ids, emb, meta))
    # upsert/insert these records to pinecone
    _ = index.upsert(vectors=to_upsert)

logger.info(f"check that we have all vectors in index : \n {index.describe_index_stats()}") 4


# load bart tokenizer and model from huggingface
tokenizer = BartTokenizer.from_pretrained('vblagoje/bart_lfqa')
generator = BartForConditionalGeneration.from_pretrained('vblagoje/bart_lfqa').to(device)

#formatting the query and generating the answer

def query_pinecone(query, top_k):
    # generate embeddings for the query
    xq = retriever.encode([query]).tolist()
    # search pinecone index for context passage with the answer
    xc = index.query(xq, top_k=top_k, include_metadata=True)
    return xc

def format_query(query, context):
    # extract passage_text from Pinecone search result and add the <P> tag
    context = [f"<P> {m['metadata']['passage_text']}" for m in context]
    # concatinate all context passages
    context = " ".join(context)
    # contcatinate the query and context passages
    query = f"question: {query} context: {context}"
    return query

def generate_answer(query):
    # tokenize the query to get input_ids
    inputs = tokenizer([query], max_length=1024, return_tensors="pt").to(device)
    # use generator to predict output ids
    ids = generator.generate(inputs["input_ids"], num_beams=2, min_length=20, max_length=40)
    # use tokenizer to decode the output ids
    answer = tokenizer.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return pprint(answer)


def predict(query):
    # search pinecone index for context passage with the answer
    context = query_pinecone(query, top_k=5)
    # format the query
    query = format_query(query, context["matches"])
    # generate answer
    answer = generate_answer(query)
    return answer
