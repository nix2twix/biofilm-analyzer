import streamlit as st
from PIL import Image
import io
import os
import subprocess

st.set_page_config(page_title="Biofilm Analyzer", layout="wide")

st.markdown("""
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

# Сессия для хранения изображения
if "image_bytes" not in st.session_state:
    st.session_state["image_bytes"] = None
if "processed_image" not in st.session_state:
    st.session_state["processed_image"] = None

# ==== БЛОК 1 ====
with st.container():
    col1, col2 = st.columns([1, 4])

    with col1:
        st.markdown("## 🧪 Biofilm Analyzer")

    with col2:
        st.markdown("### ℹ️ О приложении")
        st.markdown("""
            Этот веб-инструмент предназначен для загрузки и анализа СЭМ-изображений биоплёнок.
            Поддерживаемый формат изображений: **.bmp**. Задайте параметры анализа слева,
            загрузите изображение, а затем получите обработанные результаты.
        """)

# Разделитель
st.markdown("---")

# ==== БЛОК 2 ====
col_settings, col_workspace, col_tools = st.columns([1, 3, 1])

# --- Левый блок: Настройки анализа ---
with col_settings:
    st.markdown("### ⚙️ Настройки")
    
    area_range = st.slider(
        "Диапазон площади (px)", min_value=0, max_value=6000,
        value=(500, 3000)
    )
    min_ecc = st.slider("Минимальный эксцентриситет", 0.0, 1.0, 0.5)

# --- Центральный блок: Работа с изображением ---
with col_workspace:
    st.markdown("### 🔬 Workflow")

    if st.session_state["processed_image"]:
        st.image(st.session_state["processed_image"], caption="Processing results", use_container_width=True)
    elif st.session_state["image_bytes"]:
        st.image(st.session_state["image_bytes"], caption="Loaded image", use_container_width=True)
    else:
        st.info("Изображение пока не загружено.")

# --- Правый блок: Инструменты ---
with col_tools:
    st.markdown("### 🛠 Tools")

    uploaded_file = st.file_uploader("Load SEM-image (.bmp)", type=["bmp"], key="uploader", label_visibility="collapsed")

    if uploaded_file:
        st.session_state["image_bytes"] = uploaded_file.read()
        #st.session_state["processed_image"] = None
        #st.success("✅ Изображение загружено!")

    if st.button("🧪 Сегментация СЭМ-изображения") and st.session_state.get("image_bytes"):
        with open("input_image.bmp", "wb") as f:
            f.write(st.session_state["image_bytes"])

        subprocess.run(["python", "process.py"])

        if os.path.exists("output_image.bmp"):
            with open("output_image.bmp", "rb") as f:
                st.session_state["processed_image"] = f.read()
        else:
            st.error("Обработка не удалась. Файл не найден.")

    st.button("🔍 Включить приближение (later)")
    st.button("💾 Выгрузить результаты (later)")

