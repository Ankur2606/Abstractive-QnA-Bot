from tqdm.auto import tqdm 

# tqdm is a progress bar library with good support for nested loops and Jupyter/IPython notebooks.

class data_preprocessing:
    docs = []
    def __init__(self, wiki_data):
        self.wiki_data = wiki_data
        history = wiki_data.filter(
            lambda d: d['section_title'].startswith('History')
        )

        total_doc_count = 50000

        counter = 0
        # iterate through the dataset and apply our filter
        for d in tqdm(history, total=total_doc_count):
            # extract the fields we need
            doc = {
                "article_title": d["article_title"],
                "section_title": d["section_title"],
                "passage_text": d["passage_text"]
            }
            # add the dict containing fields we need to docs list
            data_preprocessing.docs.append(doc)

            # stop iteration once we reach 50k
            if counter == total_doc_count:
                break

            # increase the counter on every iteration
            counter += 1
