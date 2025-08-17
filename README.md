# BWSI-GitGood-NLP-Capstone

## 🌟 Project Overview

Welcome to the `BWSI-GitGood-NLP-Capstone` repository! This project is the culmination of our team's work during the third week of the **BWSI (Beaver Works Summer Institute) course**, focusing on practical applications of Natural Language Processing (NLP) and computer vision.

Our goal was to **build a semantic image search system** that allows users to find images using natural language descriptions. Given a query description, the system returns images that match the description from a database. This repository contains all the code, models, and resources developed as part of this exciting capstone project.

---

## ✨ Features

* **Semantic Similarity Search:** Implements a core search mechanism that calculates the **cosine similarity** between a user's text query embedding and a database of image embeddings. Images with higher similarity scores are considered more semantically relevant to the query.

* **Zero-shot Image Retrieval:** Uses a trained neural network with one layer to perform "zero-shot" retrieval, meaning the system can find images for categories or descriptions it has not been explicitly trained on, demonstrating strong generalization capabilities.

---

## 🚀 Technologies Used

* **Python:** The primary programming language.

* **MyGrad:** Lightweight library that adds auto-differentiation to NumPy

* **NumPy:** Data manipulation and numerical operations.

* **Matplotlib:** Data and result visualization.

!(images/COCO_Data.jpg "COCO Data")
---

## 🛠️ Installation

To set up this project locally, follow these steps:

1.  **Clone the repository:**

    ```bash
    git clone [https://github.com/gemini6829/BWSI-GitGood-NLP-Capstone.git](https://github.com/gemini6829/BWSI-GitGood-NLP-Capstone.git)
    cd BWSI-GitGood-NLP-Capstone
    ```

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    ```

3.  **Activate the virtual environment:**

    * **On Windows:**

        ```bash
        .\venv\Scripts\activate
        ```

    * **On macOS/Linux:**

        ```bash
        source venv/bin/activate
        ```

4.  **Install the required dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

    *(Note: If `requirements.txt` does not exist, you'll need to create one by listing all project dependencies, or install them manually.)*

---

## 💡 Usage

Once the installation is complete, you can run the project.

**To run the semantic image search:**

```bash
python scripts/semantic_search.py --query "a dog playing in the park" --num_results 5
