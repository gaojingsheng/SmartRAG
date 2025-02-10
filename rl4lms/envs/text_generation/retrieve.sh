
python passage_retrieval.py \
    --model_name_or_path /mnt/workspace/user/gaojingsheng/LLM/retrieval/self-rag/retrieval_lm/facebook/contriever-msmarco \
    --passages "/mnt/workspace/user/gaojingsheng/LLM/retrieval/self-rag/retrieval_lm/psgs_w100.tsv" \
    --passages_embeddings "/mnt/workspace/user/gaojingsheng/LLM/retrieval/self-rag/retrieval_lm/wikipedia_embeddings/*" \
    --query "/mnt/workspace/user/gaojingsheng/LLM/retrieval/self-rag/retrieval_lm/jingsheng_query.json"  \
    --n_docs 5