# DISC-MedLLM

<div align="left">
  
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/Flmc/DISC-MedLLM)
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/FudanDISC/DICS-MedLLM/blob/main/LICENSE)
<br>
[**Demo**](http://medllm.fudan-disc.com) | [**Tech Report**](https://arxiv.org/abs/2308.14346)
</div>
  
This is the repo of DISC-MedLLM, a medical domain-specific LLM designed for conversational healthcare scenarios by [Fudan-DISC](http://fudan-disc.com) lab.

The following resources have been released:
* DISC-Med-SFT Dataset (with out behavioral preference dataset)
* Model [weights](https://huggingface.co/Flmc/DISC-MedLLM) of DISC-MedLLM

You can check this [link](http://medllm.fudan-disc.com) to try our online demo.

## Overview
The DISC-MedLLM is a large-scale domain-specific model designed for conversational healthcare scenarios. It can address a variety of your needs, including medical consultations and treatment inquiries, offering you high-quality health support services.

The DISC-MedLLM effectively bridges the gap between general language models and real-world medical consultations, as evidenced by experimental results.

Owing to our goal-oriented strategy and the framework that integrates both LLM and Human in the loop based on real-world doctor-patient dialogues and knowledge graphs, DISC-MedLLM boasts several features:

* **Knowledge-intensive and reliable**
* **Ability of multi-turn inquiry**
* **Alignment with human preferences**

<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/data_construction.png" alt="data-construction" width="85%"/>

## Demo
### Consultation
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/consultation.gif" alt="sample1" width="60%"/>

### Treatment Inquiry
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/advice.gif" alt="sample2" width="60%"/>

## Dataset
<!-- In order to align the distribution of actual doctor responses with the intended AI doctor response distribution, our dataset is constructed from five main resources: Real-world Conversations (420k), Knowledge Graph-derived Question-Answer pairs (50k), Artificially Annotated Data aligned with human preferences (2k), MedMCQA (8k), and additional general data (34k). -->

To train DISC-MedLLM, we construct a high-quality dataset called DISC-Med-SFT consisting of over 470k distinct examples derived from existing medical datasets. We adopt a goal-oriented strategy by selectively reconstructing the dataset using a few deliberately chosen sources. These data sources serve the purpose of assisting LLMs in acquiring medical domain knowledge, aligning behavioral patterns with human preferences, and capturing real-world online medical dialogue distributions.

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style> -->
<table class="tg" style="undefined;table-layout: fixed; width: 442px">
<colgroup>
<col style="width: 204.428571px">
<col style="width: 135.428571px">
<col style="width: 102.428571px">
</colgroup>
<thead>
  <tr>
    <th class="tg-9wq8" rowspan="2"><br>Dateset</th>
    <th class="tg-9wq8" rowspan="2"><br>Original Source</th>
    <th class="tg-9wq8" rowspan="2"><br>Size</th>
  </tr>
  <tr>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">Re-constructed AI Doctor-Patient Dialogue</td>
    <td class="tg-9wq8">MedDialog</td>
    <td class="tg-9wq8">400k</td>
  </tr>
  <tr>
    <td class="tg-9wq8">cMedQA2</td>
    <td class="tg-c3ow">20k</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Knowledge Graph <br>QA pairs</td>
    <td class="tg-9wq8">CMeKG</td>
    <td class="tg-9wq8">50k</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Behavior Preference<br>Dataset</td>
    <td class="tg-9wq8">Manual selection</td>
    <td class="tg-9wq8">2k</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">Others</td>
    <td class="tg-c3ow">MedMCQA</td>
    <td class="tg-c3ow">8k</td>
  </tr>
  <tr>
    <td class="tg-c3ow">MOSS-SFT</td>
    <td class="tg-c3ow">33k</td>
  </tr>
  <tr>
    <td class="tg-c3ow">Alpaca-GPT4-zh</td>
    <td class="tg-c3ow">1k</td>
  </tr>
</tbody>
</table>

<!-- | Dataset Name                       | Number of Entries | Description                     |
|------------------------------------|-------------------|--------------------------------------------|
| MedDialog                   | 400,000           | Real-world Conversations               |
| cMedQA2                       | 20,000            | Real-world Conversations          |
| CMeKG                         | 50,000            | Knowledge Graph QA pairs                  |
| Artificially Annotated Data         | 2,000             | Data aligned with human preferences       |
| MedMCQA                       | 8,000             | Medical multiple-choice QA data           |
| moss-sft-003               | 33,000            | Other                           |
| alpaca_gpt4_data_zh           | 1,000             | Other                           | -->

<br>


### Download
We have released a total of 470k training data entries, including re-constructed dialogues and knowledge graph QA pairs. You can download the dataset via the provided [link](https://huggingface.co/datasets/Flmc/DISC-Med-SFT).
<!-- ## Training
We underwent two stages of training, both using 4*A800 GPUs. In the first stage, we trained on a dataset comprising 420k adapted AI doctor-patient dialogues, 50k knowledge graph-based QA pairs (MedMCQA), and moss-sft-003. Hyperparameters: global batch size 24, learning rate 1e-5 (AdamW optimizer), 1 epoch, max sequence length 2048 tokens, warm-up steps 1800, no weight decay.

In the second stage, known as behavior fine-tuning, we combined a 2k carefully curated AI doctor-patient dialogue dataset with 1k alpaca gpt4 zh data. Hyperparameters: global batch size 8, learning rate 5e-6 (AdamW optimizer), 1 epoch, max sequence length 2048 tokens, no weight decay.

<img src="https://github.com/t3acup/DISC-MED/blob/main/images/figure3.png" alt="Training" width="75%"/> -->
<br>

## Deploy
The current version of DISC-MedLLM is derived from the [Baichuan-13B-Base](https://github.com/baichuan-inc/Baichuan-13B). You can directly download our model weights from the HuggingFace [repository](https://huggingface.co/Flmc/DISC-MedLLM), or automatically obtain them through the demo code.

Firstly, you need to install the requirments.
```shell
pip install -r requirements.txt
```

### Using through hugging face transformers
```python
>>> import torch
>>> from transformers import AutoModelForCausalLM, AutoTokenizer
>>> from transformers.generation.utils import GenerationConfig
>>> tokenizer = AutoTokenizer.from_pretrained("Flmc/DISC-MedLLM", use_fast=False, trust_remote_code=True)
>>> model = AutoModelForCausalLM.from_pretrained("Flmc/DISC-MedLLM", device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
>>> model.generation_config = GenerationConfig.from_pretrained("Flmc/DISC-MedLLM")
>>> messages = []
>>> messages.append({"role": "user", "content": "我感觉自己颈椎非常不舒服，每天睡醒都会头痛"})
>>> response = model.chat(tokenizer, messages)
>>> print(response)
```

### Run CLI Demo
```shell
python cli_demo.py
```
### Run Web Demo
```shell
streamlit run web_demo.py --server.port 8888
```

Additionally, since the current version uses Baichuan as the base model, you can refer to its [repo](https://github.com/baichuan-inc/Baichuan-13B) for deploying with int8, int4 quantized inference. However, using quantized deployment will result in performance degradation.
<br>

## Training
You can fine-tuning our model using the data same as our data schema.
Our train code is derived from [Firefly](https://github.com/yangjianxin1/Firefly) with the different data schema and dialogue format. We jsut provide the code of Full Params Fine-tuning:
```shell
deepspeed --num_gpus={num_gpus} ./train/train.py --train_args_file ./train/train_args/sft.json
```
> Please check the setup of `sft.json` before you attempt to start training.

<br>If you want to fine-tuning our model with other training code, please use the following dialogue format.
```shell
<\b><$user_token>content<$assistant_token>content<\s><$user_token>content ...
```
The `user_token` and `assistant_token` we used are `195` and `196`, respectly. Which is same as Baichuan-13b-Chat.

## Evaluation
<!-- We compare our model with three general-purpose LLMs and two conversational Chinese medical domain LLMs. Specifically, these are GPT-3.5 and GPT-4 from OpenAI, the aligned conversational version of our backbone model Baichuan-13B-Base, Baichuan-13B-Chat, and the open-source Chinese conversational medical model HuatuoGPT-13B (trained from Ziya-Llama-13B) and BianQue-2. Our evaluation approach encompasses two key dimensions: an assessment of conversational aptitude using GPT-4 as a reference judge, and a comprehensive benchmark evaluation. -->

We assess the model's performance from two perspectives to check its capability of providing accuracy answers in single-turn conversations and presenting systematical consultation in multi-turn conversations, respectively.
* Single-turn evaluation, we construct a benchmark consisting of multiple choices questions collected from three public medical datasets and evaluate the model's accuracy.
* For multi-turn evaluation, we first construct a small set of high quality consulting cases, and then employ GPT-3.5 play the role of the patient based on the cases, and chat with the model. We use GPT-4 to evaluate the model's **proactivity**, **accuracy**, **helpfulness** and **linguistic quality**.

You can see the evalution set, dialogues generated by each model and scores provided by GPT-4 in `eval/` folder.<br>

### Single-turn evaluation
We utilized the [MLEC-QA](https://github.com/Judenpech/MLEC-QA) and Western Medicine([NEEP]()) 306 multiple-choice question datasets for our evaluation.
<!-- The MLEC-QA contains questions from the China NMLEC, categorized into Clinic, Stomatology, Public Health, Traditional Chinese Medicine, and Integrated Traditional Chinese and Western Medicine. We selected 1,362 questions (10% of the test set) for evaluation. From Western Medicine 306, we used a combined 270 questions from 2020 and 2021. Our study involved both zero-shot and few-shot approaches, with examples from MLEC-QA's validation set and 2019 Western Medicine 306 questions for the few-shot samples. -->

#### Few-shot  

| Model             | MLEC-QA Clinic | MLEC-QA CWM | MLEC-QA PublicHealth | MLEC-QA Stomatology | MLEC-QA TCM | NEEP 306 | Average |
|-------------------|----------------|-------------|----------------------|---------------------|------------|----------|---------|
| GPT-3.5           | 58.63          | 45.9        | 53.51                | 51.52               | 43.47      | 44.81    | 49.64   |
| Baichuan-13b-Chat| 31.25          | 37.69       | 28.65                | 27.27               | 29.77      | 24.81    | 29.91   |
| Huatuo(13B)        | 31.85          | 25          | 32.43                | 32.95               | 26.54      | 24.44    | 28.87   |
| DISC-MedLLM        | 44.64          | 41.42       | 41.62                | 38.26               | 39.48      | 33.33    | 39.79   |

#### Zero-shot

| Model             | MLEC-QA Clinic | MLEC-QA CWM | MLEC-QA PublicHealth | MLEC-QA Stomatology | MLEC-QA TCM | NEEP 306 | Average |
|-------------------|----------------|-------------|----------------------|---------------------|------------|----------|---------|
| GPT-3.5           | 47.32          | 33.96       | 48.11                | 39.77               | 38.83      | 33.33    | 40.22   |
| Baichuan-13b-Chat| 44.05          | 43.28       | 39.92                | 31.06               | 41.42      | 32.22    | 38.66   |
| Huatuo(13B)        | 27.38          | 21.64       | 25.95                | 25.76               | 24.92      | 20.37    | 24.34   |
| DISC-MedLLM        | 44.64          | 37.31       | 35.68                | 34.85               | 41.75      | 31.11    | 37.56   |

<!-- GPT-3.5 clearly outperformed others in the multiple-choice assessment, while our model achieved a strong second place in few-shot scenarios. In zero-shot scenarios, it followed closely behind Baichuan-13B-Chat, securing the third spot. These results highlight the current priority gap in performance for conversational medical models on knowledge-intensive tests like multiple-choice questions. -->

### Multi-turn evaluation
Our evaluation procedure draws upon three distinct datasets: Chinese Medical Benchmark ([CMB-Clin](https://github.com/FreedomIntelligence/CMB)), Chinese Medical Dialogue Dataset ([CMD](https://github.com/UCSD-AI4H/Medical-Dialogue-System)), and Chinese Medical Intent Dataset ([CMID](https://github.com/IMU-MachineLearningSXD/CMID)). CMB-Clin simulates real-world consultation process, while CMD and CMID focus on the evaluation from the perspectives of departmental specialities and user intentions. <br>

<!-- Within this framework, The Evaluation of the dialogues is based on four criteria: Proactivity, Accuracy, Helpfulness, and Linguistic Quality.

1. Proactivity: The doctor can proactively and clearly request the patient to provide more information about the symptoms, physical examination results, and medical history when the information is insufficient, actively guiding the patient through the consultation process. 
2. Accuracy: The diagnosis or advice the doctor provides is accurate and has no factual errors. Conclusions are not made arbitrarily.
3. Helpfulness: The doctor's responses provide the patient with clear, instructive, and practical assistance, specifically addressing the patient's concerns.
4. Linguistic Quality: The conversation is logical. The doctor correctly understands the patient's semantics, and the expression is smooth and natural. -->

#### Results of CMB-clin:
| **Model**              | **Proactivity** | **Accuracy** | **Helpfulness** | **Linguistic Quality** | **Average** |
|------------------------|-----------------|--------------|-----------------|------------------------|-------------|
| **GPT3.5**             | 4.30            | 4.53         | 4.55            | 5.00                   | 4.60        |
| **GPT4**               | 4.15            | 4.70         | 4.75            | 4.96                   | 4.64        |
| **Baichuan-13b-Caht**  | 4.30            | 4.58         | 4.73            | 4.95                   | 4.64        |
| **BianQue-2**          | 3.97            | 4.36         | 4.37            | 4.81                   | 4.38        |
| **Huatuo(13B)**        | 4.40            | 4.62         | 4.74            | 4.96                   | 4.68        |
| **DISC-MedLLM**        | 4.64            | 4.47         | 4.66            | 4.99                   | 4.69        |

#### Results of CMD
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/cmd.png" alt="cmd" width="75%"/>

#### Results of CMID
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/cmid.png" alt="cmid" width="75%"/>



## Acknowledgement
This project wouldn't have been possible without the support and contributions of various individuals, teams, and organizations. Special thanks go to these repositories:

- [**MedDialog**](https://github.com/UCSD-AI4H/Medical-Dialogue-System)

- [**cMeKG**](https://github.com/king-yyf/CMeKG_tools)

- [**cMedQA**](https://github.com/zhangsheng93/cMedQA2)

- [**Baichuan-13B**](https://github.com/baichuan-inc/Baichuan-13B)

- [**FireFly**](https://github.com/yangjianxin1/Firefly)

Thank you also for the work that provided important assistance to the project, but limited in length.

## Delcaration
Due to the inherent limitations of language models, we cannot assure the accuracy or reliability of information generated by this model. This model is designed exclusively for research and testing by individuals and academic groups. We urge users to critically assess any information or medical advice obtained through the model's output. Blindly trusting or following such information is strongly discouraged. We disclaim responsibility for any issues, risks, or adverse consequences resulting from the model's use.

## Licenses
The use of the source code in this repository complies with the Apache 2.0 License.

## Citation
```angular2
@misc{bao2023discmedllm,
      title={DISC-MedLLM: Bridging General Large Language Models and Real-World Medical Consultation}, 
      author={Zhijie Bao and Wei Chen and Shengze Xiao and Kuang Ren and Jiaao Wu and Cheng Zhong and Jiajie Peng and Xuanjing Huang and Zhongyu Wei},
      year={2023},
      eprint={2308.14346},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
