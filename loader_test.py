from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
import pytesseract
from PIL import Image

# -------- PDF Loader --------
pdf_path = "sample.pdf"
pdf_loader = PyPDFLoader(pdf_path)
pdf_docs = pdf_loader.load()
print("PDF Content:", pdf_docs[0].page_content)

# -------- DOCX Loader --------
docx_path = "sample.docx"
docx_loader = Docx2txtLoader(docx_path)
docx_docs = docx_loader.load()
print("DOCX Content:", docx_docs[0].page_content)

# -------- Image OCR Loader --------
image_path = "test_image.png"   # change to your image file if needed
img = Image.open(image_path)
ocr_text = pytesseract.image_to_string(img)
print("Image OCR Content:", ocr_text.strip())
