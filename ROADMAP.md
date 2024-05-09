# Roadmap

## next steps (20240509)

1. Trial using Nicole's data 
    - should be able to chunk cell by cell
    - embed a list of genes for each cell using text embedding (e.g. text-embedding-ada-02)
    - look for genes expressed above a threshold (what would be reasonable?)
    - look for the highest DE genes (e.g. top 20)
    - store the embeddings in a vector database, ideally with pgvector but might work with chromadb as well
    - use Nicole's cell annotation as metadata
2. Consider building synthetic "cell genetics " data to test the sensitivity and specificity of the approach
3. Develop a strategy to download cell-specific gene expression data