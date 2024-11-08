from pprint import pprint

import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from transformers import DataCollatorWithPadding
import evaluate
import numpy as np
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer


def load_data(path):
    train_in = pd.read_csv(path + "train/in.tsv", sep="\t", engine='python', index_col=False)
    print(len(train_in))
    print("Loading train expected")
    train_expected = pd.read_csv(path + "train/expected.tsv", sep="\t", engine='python', index_col=False)
    # print(train_expected)
    train_in = pd.concat([train_in, train_expected], axis=1)

    lines = ""
    for i, line in train_in.iterrows():
        if line["text"] == "###########################":
            train_in.at[i, "text"] = lines
            lines = ""
            # print(train_in.iloc[i]["text"])
        else:
            lines = lines + line["text"] + "\n"
    # print(train_in)

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


def load_data2(path):
    print("Loading Train in")
    with open(path + "train/in.tsv", "r") as intrain_file:
        intrain = intrain_file.read()
        train_rev = intrain.split("###########################")
    print(len(train_rev))
    train_rev[0] = train_rev[0].split("text\n")[1]
    train_rev = train_rev[:-1]
    # print(train_rev)
    train_in = pd.read_csv(path + "train/in.tsv", sep="\t", engine='python', index_col=False)
    # print(train_in)
    print("Loading train expected")
    train_expected = pd.read_csv(path + "train/expected.tsv", sep="\t", engine='python', index_col=False)
    # print(train_expected)
    train_rev = pd.DataFrame(t.replace("\n", " ") for t in train_rev)
    print("train rev")
    # print(train_rev)
    exp_rev = train_expected[train_in["text"] == "###########################"]
    # print(exp_rev)
    train_in = pd.concat([train_in, train_expected], axis=1)
    train_in = train_in[train_in["text"] != "###########################"]

    for column in exp_rev.columns:
        # print(exp_rev[column])
        train_rev[column] = list(exp_rev[column])
    # print(train_rev)

    print("Loading testA_in")
    with open(path + "test-A/in.tsv", "r") as testA_file:
        intest = testA_file.read()
        testA_rev = intest.split("###########################")
    testA_rev[0] = testA_rev[0].split("text\n")[1]

    # # print(testA_in[0])

    print("Loading testB_in")
    with open(path + "test-B/in.tsv", "r") as testA_file:
        intest = testA_file.read()
        testB_rev = intest.split("###########################")
    testB_rev[0] = testB_rev[0].split("text\n")[1]

    # print(testB_in[0])
    testA_in = pd.read_csv(path + "test-A/in.tsv", sep="\t", engine='python', index_col=False)
    testA_in = testA_in[testA_in["text"] != "###########################"]

    # train_expected=pd.read_csv(path+"train\\expected.tsv")

    testB_in = pd.read_csv(path + "test-B/in.tsv", sep="\t", engine='python', index_col=False)
    testB_in = testB_in[testB_in["text"] != "###########################"]

    # train_expected=pd.read_csv(path+"train\\expected.tsv")
    return train_in, train_rev, train_expected, testA_in, testB_in, testA_rev, testB_rev


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = sigmoid(predictions)
    predictions = (predictions > 0.5).astype(int).reshape(-1)
    return clf_metrics.compute(predictions=predictions, references=labels.astype(int).reshape(-1))


# Create torch dataset
class testDataset(Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].to(trainer.model.device) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def preprocess_function(example):
    #  print(example)
    text = example['text']
    all_labels = list(example.index)
    if "text" in all_labels: all_labels.remove("text")
    if "level_0" in all_labels: all_labels.remove("level_0")
    if "index" in all_labels: all_labels.remove("index")
    #  print(all_labels)
    labels = [0. for i in range(len(classes))]
    for label in all_labels:
        label_id = class2id[label]
        if example[label]:
            labels[label_id] = 1.

    example = tokenizer(text, truncation=True, max_length=512)
    example['labels'] = labels
    # print(example)
    return example


# tokenized_dataset = dataset.map(preprocess_function)

def train2(train_in, train_expected):
    dataset = pd.concat([train_in, train_expected], axis=1)
    train(dataset)


def train(dataset, save_as):
    # data preparation
    train_len = int(0.01 * len(dataset))
    dataset_test = dataset[:train_len].reset_index()
    dataset_train = dataset[train_len:].reset_index()
    tokenized_dataset = dataset_train.apply(preprocess_function, axis=1)
    tokenized_dataset_test = dataset_test.apply(preprocess_function, axis=1)

    # print(tokenized_dataset[0])
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=len(classes),
        id2label=id2class, label2id=class2id,

        problem_type="multi_label_classification")

    training_args = TrainingArguments(
        output_dir=save_as,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=4,
        num_train_epochs=5,
        weight_decay=0.02,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"

    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset_test,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    return trainer


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    path = "experiments/poleval2024/2024-emotion-recognition-main/"
    train_in, train_expected, testA_in, testB_in, = load_data(path)

    pprint(train_in.iloc[-10])
    print(train_expected.columns)
    model_path = 'sdadas/polish-roberta-large-v2'

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    classes = [class_ for class_ in train_expected.columns if class_]
    class2id = {class_: id for id, class_ in enumerate(classes)}
    id2class = {id: class_ for class_, id in class2id.items()}

    trainer = train(train_in, "model_train_in_oping4types3")

    ####### TEST #############
    #### test A in
    # zamien na wczytywanie najlepszego modelu
    testA_tokens = tokenizer(list(testA_in["text"]), truncation=True,max_length=512, padding=True, return_tensors="pt")
    test_dataset = testDataset(testA_tokens)
    print(test_dataset)
    outputs = trainer.predict(test_dataset)
    print(outputs)
    probs = sigmoid(outputs[0])
    print(probs)
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    print(predictions)
    print(predictions == 1)
    # turn predicted id's into actual label names
    predicted_labels = []
    for prediction in predictions:
        predicted_labels.append([id2class[idx] for idx, label in enumerate(prediction) if label == 1.0])
    with open('predictionsTestA_in_oping_4t4.tsv', 'w') as file:
        for pr in predictions:
            line = ""
            for p in pr:
                line = line + str(p == 1) + "\t"
            file.write(line[:-1])
            file.write("\n")
    # print(predicted_labels)
    print("-----------")

    #### test B in
    # zamien na wczytywanie najlepszego modelu
    testB_tokens = tokenizer(list(testB_in["text"]), truncation=True,max_length=512, padding=True, return_tensors="pt")
    test_dataset = testDataset(testB_tokens)
    print(test_dataset)
    outputs = trainer.predict(test_dataset)
    print(outputs)
    probs = sigmoid(outputs[0])
    print(probs)
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs >= 0.5)] = 1
    print(predictions)
    print(predictions == 1)
    # turn predicted id's into actual label names
    predicted_labels = []
    for prediction in predictions:
        predicted_labels.append([id2class[idx] for idx, label in enumerate(prediction) if label == 1.0])
    with open('predictionsTestB_in_oping_4t4.tsv', 'w') as file:
        for pr in predictions:
            line = ""
            for p in pr:
                line = line + str(p == 1) + "\t"
            file.write(line[:-1])
            file.write("\n")
    print("-----------")
