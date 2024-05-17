import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd


def sharpened_cosine_similarity(doc1, doc2, p=3, eps=1e-9):
    # Generate embeddings for each document
    embeddings = embed([doc1, doc2])

    # Compute cosine similarity
    cos_sim = np.inner(embeddings[0], embeddings[1])

    # Apply the sharpening formula
    sharpened_sim = np.sign(cos_sim) * (np.abs(cos_sim) + eps) ** p
    return sharpened_sim

if __name__ == "__main__":
    # Load the Universal Sentence Encoder
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    test = pd.read_csv("prompts_3000.csv")
    test["similarity"] = test.apply(lambda x: sharpened_cosine_similarity(x["rewrite_prompt"], x["pred_prompts"]), axis=1)
    test.to_csv("test.csv", index=False)