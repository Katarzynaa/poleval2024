# This is a sample Python script.
import pandas as pd

# from datasets import load_dataset
#
# from datasets import load_dataset
import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer


def load_data(path):
    train_in = pd.read_csv(path + "train/in.tsv", sep="\t", engine='python', index_col=False)
    # print(train_in)
    print("Loading train expected")
    train_expected = pd.read_csv(path + "train/expected.tsv", sep="\t", engine='python', index_col=False)
    # print(train_expected)
    train_header = pd.read_csv(path + "out-header.tsv", sep="\t", engine='python', index_col=False)
    train_emo = []
    # create list of names of emotions
    for expected in train_expected.iterrows():
        em = ""
        for tr, emo in zip(expected[1], train_header):
            if tr:
                if em != "":
                    em = em + ", " + emo
                else:
                    em = emo
                print(em)
        train_emo.append(em)
    # train_in = pd.concat([train_in, train_expected], axis=1)
    train_in["emotions"] = train_emo
    lines = ""
    for i, line in train_in.iterrows():
        if line["text"] == "###########################":
            train_in.at[i, "text"] = lines
            lines = ""
            # print(train_in.iloc[i]["text"])
        else:
            lines = lines + line["text"] + "\n"
    print(train_in)
    # train_in.to_json("train_in.jsonl", orient="records")

    testA = pd.read_csv(path + "test-A/in.tsv", sep="\t", engine='python', index_col=False)
    # print(train_in)

    lines = ""
    for i, line in testA.iterrows():
        if line["text"] == "###########################":
            testA.at[i, "text"] = lines
            lines = ""
            # print(testA.iloc[i]["text"])
        else:
            lines = lines + line["text"]

    testB = pd.read_csv(path + "test-B/in.tsv", sep="\t", engine='python', index_col=False)
    # print(train_in)

    lines = ""
    for i, line in testB.iterrows():
        if line["text"] == "###########################":
            testB.at[i, "text"] = lines
            lines = ""
            # print(testB.iloc[i]["text"])
        else:
            lines = lines + line["text"]

    return train_in, train_expected, testA, testB


def prepare_data(raw_data: pd.DataFrame, filename="train_in.jsonl"):
    with (open(filename, 'a') as outfile):
        for i, row in raw_data.iterrows():
            headline = row["text"].replace("\"", "").replace("“", "").replace("\n", " ").replace("\\", "\\\\")
            sentiment = row["emotions"]
            emo_dict = {"Joy": "Radość", "Trust": "Zaufanie", "Anticipation": "Oczekiwanie", "Surprise": "Zaskoczenie",
                        "Fear": "Strach", "Sadness": "Smutek", "Disgust": "Wstręt", "Anger": "Złość",
                        "Positive": "Pozytywny", "Negative": "Negatywny", "Neutral": "Neutralny"}
            sentimentpl = [emo_dict[em.replace(" ","")] for em in sentiment.split(",")]
            sentimentpl= ", ".join(sentimentpl)
            if i < 20:
                print(headline)

            text = ("### Instrukcja: Napisz recenzję, która wyraża podane emocje: "  # "Napisz recenzję, która wyraża emocje "
                    + sentimentpl +
                    " ### Recenzja: "+ headline
                    )

            # now append entry to the jsonl file.
            outfile.write('{"text": "' + text + '"}')
            outfile.write('\n')


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "experiments/poleval2024/2024-emotion-recognition-main/"
    train_in, train_expected, testA_in, testB_in, = load_data(path)

    train_len = int(len(train_in) * 1)
    test_in = train_in[train_len:]
    train_in = train_in[:train_len]

    prepare_data(train_in)
    prepare_data(test_in, "test.jsonl")
    #
    data_files = {"train": "train_in.jsonl", "test": "test.jsonl"}
    train_in = load_dataset("json", data_files=data_files, split="train")
    # print(senentr)

    base_model_name = "meta-llama/Meta-Llama-3-8B"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    device_map = {"": 0}

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        torch_dtype='float16'
    )
    base_model.config.use_cache = False

    # More info: https://github.com/huggingface/transformers/pull/24906
    base_model.config.pretraining_tp = 1

    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    output_dir = "./llama_napisz"

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=2e-4,
        logging_steps=10,
        max_steps=3000
    )

    max_seq_length = 512

    trainer = SFTTrainer(
        model=base_model,
        train_dataset=train_in,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    trainer.train()

    import os

    output_dir = os.path.join(output_dir, "final_checkpoint")
    trainer.model.save_pretrained(output_dir)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
