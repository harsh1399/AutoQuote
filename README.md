#### To run the mapping code with a csv file - 

Mapping is a standalone file. It has a function "get_product_recommendations" that takes csv input file. The request items from the csv needs to be converted into a list after which it can be used to get recommendations.

```
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

```

This will create a file called "top-10-products.txt" in the result folder that contains top 10 recommendations for each item requested.
