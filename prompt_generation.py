from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import pandas as pd
from tqdm.auto import tqdm
import torch
import pandas as pd
from tqdm import tqdm
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import tensorflow_hub as hub

def postprocess_text(preds):
    preds.split("<start_of_turn>model")[1].split("<end_of_turn>")[0].replace("<end_of_turn>\n<eos>","")\
        .replace("<end_of_turn>","").replace("<start_of_turn>","").replace("<eos>","").replace("<bos>","").strip()\
            .replace('"','').strip()
    preds = preds.replace("Can you make this","Make this").replace("?",".").replace("Revise","Rewrite")
    preds = preds.split(":",1)[-1].strip()
    if "useruser" in preds:
        preds = preds.replace('user', '') 
    if preds[-1].isalnum():
        preds += '.' 
    else:
        preds = preds[:-1]+'.'
    return preds


def predict_gemma(model, tokenizer, test, prime):
    predictions = []
    scores = []
    pred_prompts = []
    with torch.no_grad():
        for idx, row in tqdm(test.iterrows(), total=len(test)):
            if row.original_text == row.rewritten_text:
                predictions.append("Correct grammatical errors in this text.")
                continue
            ot = " ".join(str(row.original_text).split(" ")[:512])
            rt = " ".join(str(row.rewritten_text).split(" ")[:512])
            prompt = f"Find the orginal prompt that transformed original text to new text.\n\nOriginal text: {ot}\n====\nNew text: {rt}"
            conversation = [{"role": "user", "content": prompt }]
            prompt = tokenizer.apply_chat_template(conversation, tokenize=False) + f"<start_of_turn>model\n{prime}"
            input_ids = tokenizer.encode(prompt, add_special_tokens=False, truncation=True, max_length=1536,padding=False,return_tensors="pt")
            x = model.generate(input_ids=input_ids.to(model.device), eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.eos_token_id, max_new_tokens=128, do_sample=True, early_stopping=True, num_beams=1)
            generated_text = tokenizer.decode(x[0])
            pred_prompt = postprocess_text(generated_text)
            pred_prompts.append(pred_prompt)
    return pred_prompts

def sharpened_cosine_similarity(doc1, doc2, p=3, eps=1e-9):
    # Generate embeddings for each document
    embeddings = embed([doc1, doc2])

    # Compute cosine similarity
    cos_sim = np.inner(embeddings[0], embeddings[1])

    # Apply the sharpening formula
    sharpened_sim = np.sign(cos_sim) * (np.abs(cos_sim) + eps) ** p
    return sharpened_sim

if __name__ == "__main__":
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-1.1-2b-it")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-1.1-2b-it").to(device)
    test = pd.read_csv("prompts_3000.csv")
    prime = "General prompt: Alter"
    pred_prompts = predict_gemma(model, tokenizer, test, prime)
    test["pred_prompts"] = pred_prompts
    #calculate the similarity
    test["similarity"] = test.apply(lambda x: sharpened_cosine_similarity(x["rewrite_prompt"], x["pred_prompts"]), axis=1)
    test.to_csv("test.csv", index=False)

