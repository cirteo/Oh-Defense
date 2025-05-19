import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from tqdm.contrib import tzip
import pandas as pd
from models.LLaVA import LLaVA

def main(args):

    save_path = args.save_path
    csv_path_parts = args.csv_path.split('/')
    if len(csv_path_parts) >= 3: 
        dataset = csv_path_parts[-2]
    image_path = args.image_path

    device = "cuda"

    model = LLaVA(device,model_path=args.model_path)
    model_name = model.__class__.__name__
    print("The attacked model is " + model_name)
    
    os.makedirs(f"{save_path}/{dataset}/{model_name}/", exist_ok=True)
    query_df = pd.read_csv(args.csv_path)
    print("Generating " + f"{save_path}/{dataset}/{model_name}/{args.csv_path.split('/')[-1]}")
    batch_query_text = query_df["jailbreak_query"]
    batch_image_path = [os.path.join(image_path, path) for path in query_df["image_path"]]
    print("Image loaded.")
    if os.path.exists(f"{save_path}/{dataset}/{model_name}/{args.csv_path.split('/')[-1]}"):
        df_save = pd.read_csv(f"{save_path}/{dataset}/{model_name}/{args.csv_path.split('/')[-1]}")
        batch_response = df_save["response"].to_list()
        while len(batch_response) < len(batch_image_path):
            batch_response.append(None)
    else:
        df_save = query_df
        batch_response = [None] * len(batch_image_path)



    for index, (image_path, prompt) in enumerate(tzip(batch_image_path, batch_query_text)):
        if pd.notna(batch_response[index]) & isinstance(batch_response[index], str):
            if (len(batch_response[index]) > 100) & (
                    df_save["jailbreak_query"][index] == query_df["jailbreak_query"][index]):
                if not ('sorry' in batch_response[index]):
                    continue

        response = model.inference(image_path, prompt)

        batch_response[index] = response
        if index < 5:
            print(response)
        query_df["response"] = batch_response
        if (index == 5) or ((index + 1) % 100 == 0):
            print(f"Saving{index}...")
            query_df.to_csv(f"{save_path}/{dataset}/{model_name}/{args.csv_path.split('/')[-1]}")
    query_df["response"] = batch_response
    query_df.to_csv(f"{save_path}/{dataset}/{model_name}/{args.csv_path.split('/')[-1]}")
    # Example usage

    response_df = pd.read_csv(f"{save_path}/{dataset}/{model_name}/{args.csv_path.split('/')[-1]}")
    cnt_null = response_df['response'].isnull().sum()
    if cnt_null:
        print(f"Contain {cnt_null} Null!!!")
        print(response_df['response'][response_df['response'].isnull()])
    else:
        print(f"Done, no Nulls left.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Experiment for VLM inference")
    parser.add_argument("--model_path", type=str, default="/path/to/your/model",
                        help="Path to the model")
    parser.add_argument("--image_path", type=str, default="/path/to/your/images",
                        help="Directory for image files")
    parser.add_argument("--csv_path", type=str, default="/path/to/your/dataset.csv",
                        help="Path to the dataset CSV file")
    parser.add_argument("--save_path", default="./results/attack")
    args = parser.parse_args()

    main(args)
