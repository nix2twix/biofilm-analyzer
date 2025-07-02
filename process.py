from PIL import Image, ImageEnhance
import numpy as np
import pandas as pd

def darken_and_generate_mask(input_path, output_image_path, mask_path, csv_path):
    # Открываем изображение и затемняем
    img = Image.open(input_path).convert("RGB")
    enhancer = ImageEnhance.Brightness(img)
    darker_img = enhancer.enhance(0.6)
    darker_img.save(output_image_path)

    # Создаём "фейковую" бинарную маску (тестовая)
    np_img = np.array(img.convert("L"))
    mask = (np_img > 128).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask)
    mask_img.save(mask_path)

    # Сохраняем параметры в CSV
    df = pd.DataFrame({
        "min_area": [500],
        "max_area": [3000],
        "min_eccentricity": [0.5],
        "objects_detected": [np.random.randint(10, 50)],
    })
    df.to_csv(csv_path, index=False)

if __name__ == "__main__":
    darken_and_generate_mask("input_image.bmp", "output_image.bmp", "mask.png", "result.csv")
