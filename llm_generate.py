import pandas as pd
import torch
from numpy.random import randn
from peft import AutoPeftModelForCausalLM
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import random

base_model_name = "meta-llama/Meta-Llama-3-8B"#"Voicelab/trurl-2-7b
output_dir = "llama_napisz/final_checkpoint"

tokenizer = AutoTokenizer.from_pretrained(base_model_name, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

device_map = {"": 0}
model = AutoModelForCausalLM.from_pretrained(output_dir, torch_dtype=torch.float16, device_map=device_map, )


emo_list = ["Radość", "Zaufanie", "Oczekiwanie", "Zaskoczenie",
            "Strach", "Smutek", "Wstręt", "Złość",
            "Pozytywny", "Negatywny", "Neutralny"]
#########

text = ("Napisz recenzje zawierające podane emocje. Przykład: "
        "### Instrukcja: Napisz recenzję produktu, która wyraża podane emocje: "#"Napisz recenzję, która wyraża emocje "
        "Radość, Zaufanie, Pozytywny "
        " ### Recenzja: Cena dość niska i dla amatorów wystarczy, jeżeli ktoś poszukuje sprzęt pod cięższy aparat czy kamerę polecam wyższe modele."
        )
pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=220)
result = pipe(text)

text = ("Napisz recenzję zawierającą podane emocje. Przykład: "
        "### Instrukcja: Napisz recenzję szkoły, która wyraża podane emocje: "#Napisz recenzję, która wyraża emocje "
        "Smutek, Wstręt, Negatywny"
        " ### Recenzja: Tego pana słuchac to istna meczarnia, czyta paragrafy itp."
        )
# pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=220)
result = pipe(text)
print(result)
for i in range(1000):
    e1 = random.randint(0, 7)
    e2 = random.randint(0, 7)
    e3 = random.randint(8, 10)
    text = (    "### Instrukcja:"
                "Napisz recenzję produktu, która wyraża podane emocje: "
                #" Napisz recenzję, która wyraża emocje "
                + emo_list[e1] +", "+ emo_list[e2] +", "+ emo_list[e3] +
                " ### Recenzja: ")
        # prompt = "Co to jest LLM ?"

    # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=220)
    result = pipe(text)
    generated = result[0]['generated_text']
    if i%1==0:
        print(generated)
    ans = ["False", "False", "False", "False",
               "False", "False", "False", "False",
               "False", "False", "False"]
    ans[e1]="True"
    ans[e2] = "True"
    ans[e3] = "True"
    generated=generated.split("### Recenzja: ")[1].split("### Inst")[0].replace("\n", "")

    #     for i, emo in enumerate(emo_list):
    #         # print(emo)
    #         if emo in generated:
    #             ans[i] = "True"
    #
    with open('opinion_generated_train_product3.tsv', 'a') as file:
             #line = "\t".join(ans)
         file.write(generated)
         file.write("\n")
    with open('opinion_generated_ans_product3.tsv', 'a') as file:
         line = "\t".join(ans)
         file.write(line)
         file.write("\n")


for i in range(1000):
    e1 = random.randint(0, 7)
    e2 = random.randint(0, 7)
    e3 = random.randint(8, 10)
    text = (    "### Instrukcja:"
                "Napisz recenzję lekarza, która wyraża podane emocje: "
                #" Napisz recenzję, która wyraża emocje "
                + emo_list[e1] +", "+ emo_list[e2] +", "+ emo_list[e3] +
                " ### Recenzja: ")
        # prompt = "Co to jest LLM ?"

    # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=220)
    result = pipe(text)
    generated = result[0]['generated_text']
    if i%1==0:
        print(generated)
    ans = ["False", "False", "False", "False",
               "False", "False", "False", "False",
               "False", "False", "False"]
    ans[e1]="True"
    ans[e2] = "True"
    ans[e3] = "True"
    generated=generated.split("### Recenzja: ")[1].split("### Inst")[0].replace("\n", "")

    #     for i, emo in enumerate(emo_list):
    #         # print(emo)
    #         if emo in generated:
    #             ans[i] = "True"
    #
    with open('opinion_generated_train_doctor2.tsv', 'a') as file:
             #line = "\t".join(ans)
         file.write(generated)
         file.write("\n")
    with open('opinion_generated_ans_doctor2.tsv', 'a') as file:
         line = "\t".join(ans)
         file.write(line)
         file.write("\n")



for i in range(1000):
    e1 = random.randint(0, 7)
    e2 = random.randint(0, 7)
    e3 = random.randint(8, 10)
    text = (    "### Instrukcja:"
                "Napisz recenzję szkoły, która wyraża podane emocje: "
                #" Napisz recenzję, która wyraża emocje "
                + emo_list[e1] +", "+ emo_list[e2] +", "+ emo_list[e3] +
                " ### Recenzja: ")
        # prompt = "Co to jest LLM ?"

    # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=220)
    result = pipe(text)
    generated = result[0]['generated_text']
    if i%1==0:
        print(generated)
    ans = ["False", "False", "False", "False",
               "False", "False", "False", "False",
               "False", "False", "False"]
    ans[e1]="True"
    ans[e2] = "True"
    ans[e3] = "True"
    generated=generated.split("### Recenzja: ")[1].split("### Inst")[0].replace("\n", "")

    #     for i, emo in enumerate(emo_list):
    #         # print(emo)
    #         if emo in generated:
    #             ans[i] = "True"
    #
    with open('opinion_generated_train_school2.tsv', 'a') as file:
             #line = "\t".join(ans)
         file.write(generated)
         file.write("\n")
    with open('opinion_generated_ans_school2.tsv', 'a') as file:
         line = "\t".join(ans)
         file.write(line)
         file.write("\n")



for i in range(1000):
    e1 = random.randint(0, 7)
    e2 = random.randint(0, 7)
    e3 = random.randint(8, 10)
    text = (    "### Instrukcja:"
                "Napisz recenzję hotelu, która wyraża podane emocje: "
                #" Napisz recenzję, która wyraża emocje "
                + emo_list[e1] +", "+ emo_list[e2] +", "+ emo_list[e3] +
                " ### Recenzja: ")
        # prompt = "Co to jest LLM ?"

    # pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=220)
    result = pipe(text)
    generated = result[0]['generated_text']
    if i%1==0:
        print(generated)
    ans = ["False", "False", "False", "False",
               "False", "False", "False", "False",
               "False", "False", "False"]
    ans[e1]="True"
    ans[e2] = "True"
    ans[e3] = "True"
    generated=generated.split("### Recenzja: ")[1].split("### Inst")[0].replace("\n", "")

    #     for i, emo in enumerate(emo_list):
    #         # print(emo)
    #         if emo in generated:
    #             ans[i] = "True"
    #
    with open('opinion_generated_train_hotel2.tsv', 'a') as file:
             #line = "\t".join(ans)
         file.write(generated)
         file.write("\n")
    with open('opinion_generated_ans_hotel2.tsv', 'a') as file:
         line = "\t".join(ans)
         file.write(line)
         file.write("\n")