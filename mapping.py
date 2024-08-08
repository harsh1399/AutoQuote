from sentence_transformers import SentenceTransformer
import pandas as pd
import faiss
import numpy as np
from huggingface_hub import login
import os
# login(token = os.getenv("HUGGINGFACE_TOKEN"))
model = SentenceTransformer("Snowflake/snowflake-arctic-embed-m-v1.5")  #sentence-transformers/all-MiniLM-L6-v2
product_db = pd.read_csv("data/productDB.csv")
if not os.path.isfile("data/product_embeddings.index"):
    desc1 = product_db.iloc[:, 2].tolist()
    desc2 = product_db.iloc[:, 3].tolist()
    # desc3 = product_db.iloc[:, 5].tolist()
    # desc4 = product_db.iloc[:, 6].tolist()
    # desc5 = product_db.iloc[:, 7].tolist()
    product_descriptions = [f"{des1}, {des2}" for des1, des2 in
                            zip(desc1, desc2)]
    print(product_descriptions)
    product_embeddings = model.encode(product_descriptions)
    dimension = product_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(np.array(product_embeddings))
    faiss.write_index(index,"data/product_embeddings.index")

index = faiss.read_index('data/product_embeddings.index')

def faiss_filter_products(request_items,k=20):
    with open("result/requested_items.txt",'w') as f:
        f.writelines([item+'\n' for item in request_items])
    request_items_embeddings = model.encode(request_items)
    distances, indices = index.search(np.array(request_items_embeddings),k)
    results = []
    for i, item in enumerate(request_items):
        top_n_descriptions = product_db.iloc[indices[i],:]
        # print(top_n_descriptions)
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
        final_result = [f"{desc1} {desc2} {desc3} {desc4} {desc5} {desc6} {desc7} {desc8} {desc9} {desc10}\n" for desc1, desc2, desc3, desc4, desc5, desc6, desc7,desc8,desc9,desc10 in zip(description1,description2,description3,description4,description5,description6,description7,description8,description9,description10)]
        with open("result/top-20-recommendations.txt",'a') as f:
            f.write(f"-------{item}--------\n")
            f.writelines(top_n_descriptions)
        results.append(top_n_descriptions)
