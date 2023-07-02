import streamlit as st
from PIL import Image
import io
import os
import replicate
import numpy as np


def traitement(image, prompt):

    image_bytes = io.BytesIO()
    image = image.convert('RGB')
    image.save(image_bytes, format='JPEG')  # Vous pouvez spécifier le format souhaité (JPEG, PNG, etc.)

    image_bytes.seek(0)

    model = replicate.Client(api_token = st.secrets["REPLICATE_API_KEY"])

    output = model.run(
        "daanelson/minigpt-4:b96a2f33cc8e4b0aa23eacfce731b9c41a7d9466d9ed4e167375587b54db9423",
        input={"image": image_bytes, "prompt":prompt},
    )
    print(output)
    return output

st.title("🩻 Delos Image Describer")

uploaded_file = st.file_uploader("Drop une image ici", type=["jpg", "png", "jpeg"])
prompt = st.text_input("Entrez votre question ici")

# Afficher l'image dès qu'elle est téléchargée
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Créer deux colonnes, une pour l'image et une pour le prompt
    col1, col2 = st.columns([1, 2])

    # Afficher l'image dans la première colonne, avec une largeur personnalisée
    col1.image(image, caption='Image uploadée.', use_column_width=True, width=150)

# Exécuter le traitement seulement après l'entrée d'un prompt
if uploaded_file is not None and prompt:
    # Appeler la fonction de traitement ici...
    response = traitement(image, prompt)

    # Afficher le résultat dans la deuxième colonne
    col2.write(response)
else:
    st.write("Veuillez uploader une image et entrer un prompt.")
