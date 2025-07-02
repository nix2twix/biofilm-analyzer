from PIL import Image, ImageEnhance

def darken_image(input_path, output_path, factor=0.6):
    img = Image.open(input_path).convert("RGB")
    enhancer = ImageEnhance.Brightness(img)
    darker_img = enhancer.enhance(factor)
    darker_img.save(output_path)

if __name__ == "__main__":
    darken_image("input_image.bmp", "output_image.bmp")
