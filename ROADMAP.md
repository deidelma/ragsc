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

## Plan (20240511)

1. implement basic analysis pipeline as described in the Scanpy tutorial
2. Import the data set that Nicole used in her first analysis
3. create an initial database using a subset of the cells
   1. for each cell calculate the genes with the largest differential expression (start with n=20)
   2. create a string containing the gene names in order of expression from highest to lowest
   3. embed the gene expression in a vector
   4. store the vectors in pgvector database with cell type as metadata
   5. create a set of cells of the same nominal cell type from the original dataset to explore the best similarity algorithm
4. If the above is successful, attempt to create a database using all the data from the first paper
5. Using the above attempt to identify cells from another paper being used by Nicole