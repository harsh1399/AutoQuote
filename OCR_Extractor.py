from PIL import Image
import pytesseract
import fitz
import os
import shutil

def convert_to_image(input_file):
    """
    Convert pages of pdf file into images and stores them in data/images directory.
    :param input_file: PDF file uploaded on streamlit
    :return: None
    """
    delete_all_files_in_folder("data/images")
    doc = fitz.open(stream=input_file.read(),filetype="pdf")
    for page_index in range(len(doc)):
        page = doc.load_page(page_index)
        mat = fitz.Matrix(2.0,2.0)
        pix = page.get_pixmap(matrix = mat)
        image_path = f"data/images/{page_index}-image.png"
        pix.save(image_path)
        im = Image.open(f"data/images/{page_index}-image.png")
        im.save(f"data/images/{page_index}-image.png", dpi=(600, 600))

def image_to_text(image_path):
    """
    Extract text from images. The system must have tesseract installed to utilize this function.
    :param image_path: path of the image from which text needs to be extracted
    :return: text: from the image
    """
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img)
    return text

def delete_all_files_in_folder(folder_path):
    """
    Delete all the file in the images folder once user uploads new document on streamlit application.
    :param folder_path: path of the images folder
    :return: None
    """
    if os.path.exists(folder_path) and os.path.isdir(folder_path):
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # remove the file or symbolic link
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # remove the directory and all its contents
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
        # print(f"All files in {folder_path} have been deleted.")
    else:
        print(f"{folder_path} is not a valid directory.")




