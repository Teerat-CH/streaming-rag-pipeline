import random
from pathlib import Path

import ir_datasets
dataset = ir_datasets.load("msmarco-document/dev")
random.seed(200)

# test
# for query in dataset.queries_iter():
#     print(query)      
#     break

# for qrel in dataset.qrels_iter():
#     print(qrel)       
#     break

# qrels
# namedtuple<query_id, doc_id, relevance, iteration>
qrels = list(dataset.qrels_iter())[:200]

# queries based on the query_id from qrels
# namedtuple<query_id, text>
query_ids = {q.query_id for q in qrels}
queries = [q for q in dataset.queries_iter() if q.query_id in query_ids]

# docs based on the doc_id from qrels
# dict: doc_id -> namedtuple<doc_id, url, title, body>
doc_ids = {q.doc_id for q in qrels}
store = dataset.docs_store()
docs = store.get_many(doc_ids)

# 200 random extra docs with doc_id not in doc_ids
# dict: doc_id -> namedtuple<doc_id, url, title, body>
n_docs = store.count()
all_docs = dataset.docs_iter()
extra_docs = {}
while len(extra_docs) < 200:
    doc = all_docs[random.randrange(n_docs)]
    if doc.doc_id not in doc_ids:
        extra_docs[doc.doc_id] = doc

# docs + extra_docs
# dict: doc_id -> namedtuple<doc_id, url, title, body>
mini_docs = docs.copy()
mini_docs.update(extra_docs)

# # print
# doc = next(iter(mini_docs.values()))
# print(repr(doc.body[:300]))

# data cleaning and write to tsv
def write_tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for row in rows:
            fields = [str(x).replace("\t", " ").replace("\n", " ").replace("\r", " ") for x in row]
            f.write("\t".join(fields) + "\n")

out = Path(__file__).parent
write_tsv(out / "mini_qrels.tsv", ["query_id", "doc_id", "relevance"],
          [(q.query_id, q.doc_id, q.relevance) for q in qrels])
write_tsv(out / "mini_queries.tsv", ["query_id", "text"], queries)
write_tsv(out / "mini_docs.tsv", ["doc_id", "url", "title", "body"],
          [mini_docs[d] for d in sorted(mini_docs)])