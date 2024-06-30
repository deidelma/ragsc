# Roadmap

## 2024-06-30

need to create a loop that

* load the base data
* loop across expression levels (0, 0.5, 1.0, 1.5, 2.0, 2.5)
    * calculate signatures based on expression level
    * eliminate "redundant genes" from signatures
    * calculate the embeddings for the signatures
    * divide the data into train and target
    * store the train data into vector database
    * compare the original cluster number to the top 5 predicted cluster numbers 
        * graph the results