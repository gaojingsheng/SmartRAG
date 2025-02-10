# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys

import argparse
import json
import pickle
import time
import glob
from pathlib import Path

import numpy as np
import torch
import transformers

sys.path.append("/cpfs/user/gaojingsheng/AliA800/self-rag/retrieval_lm")
import src.index
import src.contriever
import src.utils
import src.slurm
import src.data
from src.evaluation import calculate_matches
import src.normalize_text

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class Retriever:
    def __init__(self, model=None, tokenizer=None):
        self.model = model
        self.tokenizer = tokenizer
        self.passages = "/cpfs/user/gaojingsheng/AliA800/self-rag/retrieval_lm/psgs_w100.tsv"
        self.passages_embeddings = "/cpfs/user/gaojingsheng/AliA800/self-rag/retrieval_lm/wikipedia_embeddings/*"
        # self.passages_embeddings = "/cpfs/user/gaojingsheng/AliA800/self-rag/retrieval_lm/wikipedia_embeddings/passages_00"
        self.n_docs = 1
        self.validation_workers = 64
        self.per_gpu_batch_size = 64
        self.save_or_load_index = False
        self.model_name_or_path = "/cpfs/user/gaojingsheng/AliA800/self-rag/retrieval_lm/facebook/contriever-msmarco"
        self.no_fp16 = True
        self.question_maxlength = 512
        self.indexing_batch_size = 1000000
        self.projection_size = 768
        self.n_subquantizers = 0
        self.n_bits = 8
        self.lowercase = False
        self.normalize_text = False
        # parser = argparse.ArgumentParser()
        # self.args = parser.parse_args()
        # src.slurm.init_distributed_mode(self.args)
        
    def embed_queries(self, queries):
        # print("queries is ", queries)
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                if self.lowercase:
                    q = q.lower()
                if self.normalize_text:
                    q = src.normalize_text.normalize(q)
                batch_question.append(q)

                if len(batch_question) == self.per_gpu_batch_size or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=self.question_maxlength,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        # print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()
    

    def embed_queries_demo(self, queries):
        embeddings, batch_question = [], []
        with torch.no_grad():
            for k, q in enumerate(queries):
                batch_question.append(q)

                if len(batch_question) == 16 or k == len(queries) - 1:

                    encoded_batch = self.tokenizer.batch_encode_plus(
                        batch_question,
                        return_tensors="pt",
                        max_length=200,
                        padding=True,
                        truncation=True,
                    )
                    encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                    output = self.model(**encoded_batch)
                    embeddings.append(output.cpu())

                    batch_question = []

        embeddings = torch.cat(embeddings, dim=0)
        print(f"Questions embeddings shape: {embeddings.size()}")

        return embeddings.numpy()

    def index_encoded_data(self, index, embedding_files, indexing_batch_size):
        allids = []
        allembeddings = np.array([])
        for i, file_path in enumerate(embedding_files):
            print(f"Loading file {file_path}")
            with open(file_path, "rb") as fin:
                ids, embeddings = pickle.load(fin)

            allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
            allids.extend(ids)
            while allembeddings.shape[0] > indexing_batch_size:
                allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        while allembeddings.shape[0] > 0:
            allembeddings, allids = self.add_embeddings(index, allembeddings, allids, indexing_batch_size)

        print("Data indexing completed.")

    def add_embeddings(self, index, embeddings, ids, indexing_batch_size):
        end_idx = min(indexing_batch_size, embeddings.shape[0])
        ids_toadd = ids[:end_idx]
        embeddings_toadd = embeddings[:end_idx]
        ids = ids[end_idx:]
        embeddings = embeddings[end_idx:]
        index.index_data(ids_toadd, embeddings_toadd)
        return embeddings, ids


    def add_passages(self, passages, top_passages_and_scores):
        # add passages to original data
        docs = [passages[doc_id] for doc_id in top_passages_and_scores[0][0]]
        return docs

    def setup_retriever(self):
        print(f"Loading model from: {self.model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(self.model_name_or_path)
        self.model.eval()
        self.model = self.model.cuda()
        if not self.no_fp16:
            self.model = self.model.half()
        self.index = src.index.Indexer(self.projection_size, self.n_subquantizers, self.n_bits)

        # index all passages
        input_paths = glob.glob(self.passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if self.save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, self.indexing_batch_size)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")
            if self.save_or_load_index:
                self.index.serialize(embeddings_dir)

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(self.passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

    def search_document(self, query, top_n=10):
        questions_embedding = self.embed_queries([query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, self.n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:top_n]
    
    def search_document_demo(self, query, n_docs=10):
        questions_embedding = self.embed_queries_demo([query])

        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = self.index.search_knn(questions_embedding, n_docs)
        print(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        return self.add_passages(self.passage_id_map, top_ids_and_scores)[:n_docs]

    def setup_retriever_demo(self, model_name_or_path, passages, passages_embeddings, n_docs=5, save_or_load_index=False):
        print(f"Loading model from: {model_name_or_path}")
        self.model, self.tokenizer, _ = src.contriever.load_retriever(model_name_or_path)
        self.model.eval()
        self.model = self.model.cuda()

        self.index = src.index.Indexer(768, 0, 8)

        # index all passages
        input_paths = glob.glob(passages_embeddings)
        input_paths = sorted(input_paths)
        embeddings_dir = os.path.dirname(input_paths[0])
        index_path = os.path.join(embeddings_dir, "index.faiss")
        if save_or_load_index and os.path.exists(index_path):
            self.index.deserialize_from(embeddings_dir)
        else:
            print(f"Indexing passages from files {input_paths}")
            start_time_indexing = time.time()
            self.index_encoded_data(self.index, input_paths, 1000000)
            print(f"Indexing time: {time.time()-start_time_indexing:.1f} s.")

        # load passages
        print("loading passages")
        self.passages = src.data.load_passages(passages)
        self.passage_id_map = {x["id"]: x for x in self.passages}
        print("passages have been loaded")

def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    return data



if __name__ == "__main__":

    retriever = Retriever()
    retriever.setup_retriever()
    output = retriever.search_document("Who is the president of America?", 1)
    print(output)
    
    print(output)
    print(output[0]['title'])
    print(output[0]['text'])