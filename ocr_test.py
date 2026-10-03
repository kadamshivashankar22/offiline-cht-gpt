import pytesseract
from PIL import Image

# Create a test image with text
img = Image.new('RGB', (300, 100), color=(255, 255, 255))
from PIL import ImageDraw
d = ImageDraw.Draw(img)
d.text((10, 40), "Hello OCR Test!", fill=(0, 0, 0))
img.save("test_image.png")

# Load and run OCR
text = pytesseract.image_to_string(Image.open("test_image.png"))
print("OCR Output:", text.strip())
