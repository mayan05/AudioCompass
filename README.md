# 🎵 AudioCompass

An **ML-powered audio analyzer** and Streamlit app that helps you understand audio at a glance.

---

## 📋 Objective

To develop a machine learning–powered audio analyzer that can accurately:

- 🎼 Detect the **musical key & scale** (e.g., C Major, A♯ Minor)
- 🎵 Estimate the **tempo (BPM)** of a track

…all from a single audio file using a **shared feature learning model**, deployed as an interactive **Streamlit application**.

---

## 🧩 Problem Formulation

- **Input:**  
  An audio file (e.g., `.wav`, `.mp3`)

- **Output:**
  - **Key:** One of 12 tonal keys (`C`, `C#`, `D`, …, `B`)
  - **Scale:** `Major` / `Minor`
  - **BPM:** A continuous numerical value (regression)

---

## 🚀 Features

✅ Accepts `.wav` or `.mp3` audio input  
✅ Predicts:
  1. 🎼 Musical **key & scale**
  2. 🎵 **Tempo (BPM)**

---

## 📊 Results & Visualizations

Here are some example results from the model evaluation and testing:

### 🔷 Graph 1
![Graph 1](readme_graphs/graph_1.png)

### 🔷 Graph 2
![Graph 2](readme_graphs/graph_2.png)

### 🔷 Graph 3
![Graph 3](readme_graphs/graph_3.png)

### 🔷 Graph 4
![Graph 4](readme_graphs/graph_4.png)

---

## 📂 How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

### 2️⃣ Run the app
```bash
streamlit run app.py
```

### 3️⃣ Upload an audio file and view predictions.

---

## 🛠️ Technologies Used
-**PyTorch🔥**-

-**Streamlit 📈**-

-**Librosa 🎧**-

-**Scikit-learn 🔬**-

-**FastAPI 🍃**-

-**Numpy 🔢**-

---

## 📄 License
MIT License © 2025
