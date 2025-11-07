import tkinter as tk
from tkinter import messagebox
import numpy as np
from utils import *
import pickle

from tensorflow.keras.models import load_model

# Load your saved model and tokenizer
model = load_model("lstm_twitter.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 23

# Prediction function
def predict_text():
    text = entry.get("1.0", tk.END).strip()
    if not text:
        messagebox.showwarning("Input Error", "Please enter some text.")
        return
    
    text=clean_text(text)
    
    class_name=get_pred(model,tokenizer,[text],MAX_LEN)
    
    # Show result
    result_label.config(text=f"Predicted Class: {class_name}", fg="blue")

# Create GUI window
root = tk.Tk()
root.title("Text Classification App")
root.geometry("500x350")
root.config(bg="#f5f5f5")

# Title
title_label = tk.Label(root, text="Text Classification Model", font=("Arial", 16, "bold"), bg="#f5f5f5")
title_label.pack(pady=10)

# Text input
entry_label = tk.Label(root, text="Enter your text below:", bg="#f5f5f5")
entry_label.pack()
entry = tk.Text(root, height=6, width=50, font=("Arial", 12))
entry.pack(pady=10)

# Predict button
predict_btn = tk.Button(root, text="Predict", command=predict_text, bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
predict_btn.pack(pady=10)

# Output label
result_label = tk.Label(root, text="", font=("Arial", 14), bg="#f5f5f5")
result_label.pack(pady=10)

# Run GUI
root.mainloop()
