# Land Cover Prediction Streamlit App

This app predicts land cover classes from satellite images using a BigEarthNet pretrained model from Hugging Face.

## Features
- Upload satellite images (RGB, 224x224 or larger, jpg/png/tif)
- Predict multi-label land cover classes with probabilities
- Display uploaded image and results
- Optional: Show Folium map for context
- Save/delete results and visualizations
##Adavanced (soon)
Customization & Advanced Features
If you want to use a different model (e.g., with “wildfire prone” or “land fill area” classes), just change the MODEL_NAME in app.py to your preferred Hugging Face model.
The app is structured for easy extension: you can add graphs, heatmaps, and more advanced UI features as needed.
The save/delete/download logic is started; you can expand it to save results in more formats or locations.
Would you like help with:
Integrating a specific model with “wildfire prone” or “land fill area” classes?
Adding advanced visualizations (graphs, heatmaps)?
Improving the UI/UX or navigation?
Anything else? Let me know your next priority!

## Usage
1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the app:
   ```bash
   streamlit run app.py
   ```
3. Open the provided local URL in your browser.

## Deployment
- Deploy to [Streamlit Cloud](https://streamlit.io/cloud) or [Hugging Face Spaces](https://huggingface.co/spaces) for a public URL.

## Model
- Uses a BigEarthNet pretrained model from Hugging Face (e.g., `qubvel/efficientnet-b0_bigearthnet`).
- You can change the model in `app.py` as needed. 