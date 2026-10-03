from reportlab.pdfgen import canvas

file_path = "sample.pdf"
c = canvas.Canvas(file_path)
c.drawString(100, 750, "Hello! This is a sample PDF created for testing.")
c.drawString(100, 730, "You can now load this with PyPDFLoader.")
c.save()

print(f"PDF created: {file_path}")
