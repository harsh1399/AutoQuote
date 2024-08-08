from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd
import faiss
import numpy as np
from huggingface_hub import login
import os

## Bi-encoder model to retrieve top 20 results from FAISS database
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5",model_kwargs=dict(add_pooling_layer = False))  #sentence-transformers/all-MiniLM-L6-v2

## Cross-encoder to rerank the results that we recieve from biencoder model
cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-12-v2',revision=None)


product_db = pd.read_csv("data/productDB.csv")

## Create and store embeddings of productDB.csv in a FAISS vector database.
if not os.path.isfile("data/product_embeddings.index"):
    desc1 = product_db.iloc[:, 2].tolist()
    desc2 = product_db.iloc[:, 3].tolist()
    product_descriptions = [f"{des1}, {des2}" for des1, des2 in
                            zip(desc1, desc2)]
    product_embeddings = model.encode(product_descriptions)
    dimension = product_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(product_embeddings))
    faiss.write_index(index,"data/product_embeddings.index")

index = faiss.read_index('data/product_embeddings.index')

def faiss_filter_products(request_items,k=20):
    """
    Filter products from FAISS database based on the requested items by the user. Filter is based on semantic search.
    :param request_items: list of items requested by the user
    :param k: Top k item will be retrieved
    :return: None
    """
    with open("result/requested_items.txt",'w') as f:
        f.writelines([item+'\n' for item in request_items])
    request_items_embeddings = model.encode(request_items,prompt_name="query")
    distances, indices = index.search(np.array(request_items_embeddings),k)
    results = []
    for i, item in enumerate(request_items):
        top_n_descriptions = product_db.iloc[indices[i],:]
        description1 = top_n_descriptions.iloc[:,0]
        description2= top_n_descriptions.iloc[:,1]
        description3 = top_n_descriptions.iloc[:,2]
        description4 = top_n_descriptions.iloc[:,3]
        description5 = top_n_descriptions.iloc[:, 4]
        description6 = top_n_descriptions.iloc[:, 5]
        description7 = top_n_descriptions.iloc[:, 6]
        description8 = top_n_descriptions.iloc[:, 7]
        description9 = top_n_descriptions.iloc[:, 9]
        description10 = top_n_descriptions.iloc[:, 10]
        cross_enc_desc = [f"{desc1} {desc2}" for desc1,desc2 in zip(description3,description4)]
        top_20_string = [f"{desc1} {desc2} {desc3} {desc4} {desc5} {desc6} {desc7} {desc8} {desc9} {desc10}\n" for desc1, desc2, desc3, desc4, desc5, desc6, desc7,desc8,desc9,desc10 in zip(description1,description2,description3,description4,description5,description6,description7,description8,description9,description10)]
        with open("result/top-20-recommendations.txt",'a') as f:
            f.write(f"\n-------{item}--------\n")
            f.writelines(top_20_string)
        rerank_products(item,cross_enc_desc,top_20_string)

def rerank_products(request_item,top_n_descriptions,top_20_string):
    """
    Use the cross encoder to rerank the products.
    :param request_item: list of items requested by the user
    :param top_n_descriptions: description of top 20 products
    :param top_20_string: complete description of top 20 products.
    :return: None
    """
    pairs = [[request_item,desc] for desc in top_n_descriptions]
    scores = cross_encoder.predict(pairs)
    sorted_indices = np.argsort(scores)[::-1][:10]
    with open("result/top-10-products.txt",'a') as f:
        f.write(f"\n---------{request_item}---------\n")
        for idx in sorted_indices:
            f.write(top_20_string[idx])

def get_product_recommendations(input_file,k=20):
    """
    generate product recommendations from csv
    :param input_file: csv file containing product recommendations
    :param k: top k recommendations will be retrieved from biencoder
    :return: None
    """

    order_request = pd.read_csv(input_file)
    #####################
    # Convert request_items into a list

    # request_items = order_request.iloc[:,0].tolist()
    ###################

    ## uncomment the below code after converting into a list of items

    #faiss_filter_products(request_items,k)
    """
    This will create a file called "top-10-products.txt" that contains top 10 recommendations for each item requested.
    """
