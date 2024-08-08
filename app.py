from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import os
import OCR_Extractor
import google.generativeai as genai
import pathlib
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import mapping

genai.configure(api_key = os.getenv("GOOGLE_API_KEY"))
model = genai.GenerativeModel('gemini-1.5-flash',
                          system_instruction="You are an expert in understanding and processing order requests. The user will provide an input image containing order requests for various items. Analyze the input image. Your task is to extract the item descriptions for each request with precision.")

def create_prompt():
    return """For the given image, extract the item descriptions/materials ordered by the customer. The order requests are generally presented in structured tabular formats with proper headings. Only return the list of items ordered by the customer, with each item on a new line. Do not provide anything extra (like quantity,price per, etc.) in the answer.
    Here's an example for your reference - 
    Although, the example is not an image, you will find similar structure in the images as well.
    Data from image - 
    Requesting contractor: Harsh Mahajan              Job Site: Salt Lake City, Utah
    Note to supplier:
    Expected ship date:                               Date:
     No      |                      Description                        |   Quantity 
     1       |             [CB] Corner Bead - paper faced 10'          |    35,288.66
     2       |             [T35820] 3-5/8" 20g a Track, 10'            |    67,703.01
     3       |             [S35820] 3-5/8" 20g a Stud                  |    235,430.39
     4       |       [T35820SLOTTED] 3-5/8" 20ga Sloted track, 10'     |    29,362.03
     5       |                  [X58] 58/" Firecode Core               |    1,114,504.66
    
    Answer returned: ["[CB] Corner Bead - paper faced 10'",
    "[T35820] 3-5/8" 20g a Track, 10'",
    "[S35820] 3-5/8" 20g a Stud",
    "[T35820SLOTTED] 3-5/8" 20ga Sloted track, 10'",
    "[X58] 58/" Firecode Core"]
    Now, for the given image, extract the items requested by the customer - 
    Items requested:
    """

def generate_response(prompt):
    try:
        response = model.generate_content([prompt[0],prompt[1]])
        return response.text
    except Exception as e:
        time.sleep(60)
        try:
            response = model.generate_content([prompt[0], prompt[1]])
            return response.text
        except Exception as e1:
            print(e1)
            return "Error by gemini"

def get_results(prompts, max_workers = 3):
    with ThreadPoolExecutor() as executor:
        executor._max_workers = max_workers
        results = list(executor.map(generate_response,prompts))
    return results

def convert_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        OCR_Extractor.convert_to_image(uploaded_file)

st.set_page_config(page_title="Quotation Modelling")
st.header("quotation automation")
uploaded_file = st.file_uploader("Upload a PDF",type=["pdf"])

submit = st.button("Get Quotation")
if submit:
    convert_uploaded_file(uploaded_file)
    files = os.listdir("data/images")
    prompts = []
    for file in files:
        file_extension = pathlib.Path(f"data/images/{file}").suffix
        if file_extension == ".png":
            prompt = create_prompt()
            image_file = Image.open(f"data/images/{file}")
            # items = generate_response(prompt,image_file)
            prompts.append([prompt,image_file])
    result = get_results(prompts)
    time.sleep(30)
    extracted_items = []
    for res in result:
        extracted_items.extend(res.split("\n"))
    mapping.faiss_filter_products(extracted_items)


