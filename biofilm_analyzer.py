import streamlit as st
import subprocess
import os
import json
import sys
# ============== Настройки страницы ==============
st.set_page_config(page_title="Biofilm Analyzer", layout="wide")

# ============== Состояния сессии ==============
if "image_bytes" not in st.session_state:
    st.session_state.image_bytes = None
if "processed_image" not in st.session_state:
    st.session_state.processed_image = None
if "area_range" not in st.session_state:
    st.session_state.area_range = (500, 3000)
if "min_ecc" not in st.session_state:
    st.session_state.min_ecc = 0.5
if "trigger_processing" not in st.session_state:
    st.session_state.trigger_processing = False

# ============== Блок 1: Заголовок и описание ==============
with st.container():
    col1, col2 = st.columns([1, 4])

    with col1:
        st.markdown("## 🧪 Biofilm Analyzer")

    with col2:
        st.markdown("### ℹ️ Info")
        st.markdown("""
This tool is designed for processing SEM images of biofilms. The supported image format is .bmp, .png, and .jpg. 
Set the analysis parameters on the left, upload the image, and get the processing result.
        """)

st.markdown("---")

# ============== Блок 2: Интерфейс ==============
col_settings, col_workspace, col_tools = st.columns([1, 3, 1])

# === Левая панель: Settings ===
with col_settings:
    st.markdown("### ⚙️ Settings")

    area_range = st.slider(
        "Area range (px)",
        min_value=0, max_value=6000,
        value=st.session_state.area_range,
        key="area_slider"
    )

    min_ecc = st.slider(
        "Minimum eccentricity",
        min_value=0.0, max_value=1.0,
        value=st.session_state.min_ecc,
        key="ecc_slider"
    )

    if area_range != st.session_state.area_range or min_ecc != st.session_state.min_ecc:
        st.session_state.area_range = area_range
        st.session_state.min_ecc = min_ecc
        if st.session_state.image_bytes:
            st.session_state.trigger_processing = True

# === Центральная панель: Workflow ===
with col_workspace:
    st.markdown("### 🔬 Workflow")

    if st.session_state.processed_image:
        st.image(st.session_state.processed_image, caption="Processing result", use_container_width=True)
    elif st.session_state.image_bytes:
        st.image(st.session_state.image_bytes, caption="Loaded SEM-image", use_container_width=True)
    else:
        st.info("SEM-image wasn't uploaded")

# === Правая панель: Инструменты ===
# --- Инициализация состояний ---
for key, value in {
    "image_bytes": None,
    "processed_image": None,
    "area_range": (500, 3000),
    "min_ecc": 0.5,
    "image_uploaded": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = value

# --- Правая панель: Инструменты ---
with col_tools:
    st.markdown("### 🛠 Tools")

    # --- Загрузка изображения ---
    uploaded_file = st.file_uploader("Load image (.bmp)", type=["bmp"], key="uploader")

    # --- Загрузка нового изображения ---
    if uploaded_file is not None and not st.session_state.image_uploaded:
        st.session_state.image_bytes = uploaded_file.read()
        st.session_state.processed_image = None
        st.session_state.image_uploaded = True
        st.rerun()

    # --- Сброс изображения ---
    elif uploaded_file is None and st.session_state.image_uploaded:
        st.session_state.image_bytes = None
        st.session_state.processed_image = None
        st.session_state.image_uploaded = False

    # --- Инструменты ---
    seg_button_clicked = st.button("🧪 Start segmentation", disabled=st.session_state.image_bytes is None)
    st.button("🔍 Zoom (see later)")
    st.button("💾 Save results (see later)")

    if seg_button_clicked:
        with st.spinner("⏳ Image processing..."):
            # Сохраняем изображение
            with open("input_image.bmp", "wb") as f:
                f.write(st.session_state.image_bytes)

            # Сохраняем параметры (на будущее)
            params = {
                "min_area": st.session_state.area_range[0],
                "max_area": st.session_state.area_range[1],
                "min_eccentricity": st.session_state.min_ecc
            }
            with open("params.json", "w") as f:
                json.dump(params, f)

            # Обработка изображения
            result = subprocess.run([sys.executable, "process.py"])

            if result.returncode != 0:
                st.error("❌ Error while processing image")
                st.text(result.stdout)
                st.text(result.stderr)
            elif os.path.exists("output_image.bmp"):
                with open("output_image.bmp", "rb") as f:
                    st.session_state.processed_image = f.read()
                st.rerun()
                st.success("✅ Processed successfully")
            else:
                st.rerun()
                st.error("❌ No result file was found")

