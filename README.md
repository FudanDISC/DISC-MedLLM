# DISC-MedLLM

<div align="left">
  
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/Flmc/DISC-MedLLM)
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/FudanDISC/DICS-MedLLM/blob/main/LICENSE)

</div>
  
This is the repo of DISC-MedLLM, a medical domain-specific LLM designed for conversational healthcare scenarios by [Fudan-DISC](http://fudan-disc.com) lab.

The following resources have been released:
* DISC-Med-SFT (with out behavioral preference dataset)
* Model [weight](https://huggingface.co/Flmc/DISC-MedLLM) of DISC-MedLLM

You can check this [link](http://medllm.fudan-disc.com) to try our online demo.

## Overview
The DISC-MedLLM is a large-scale domain-specific model designed for conversational healthcare scenarios. It can address a variety of your needs, including medical consultations and treatment inquiries, offering you high-quality health support services.

The DISC-MedLLM effectively bridges the gap between general language models and real-world medical consultations, as evidenced by experimental results.

Owing to our goal-oriented strategy and the framework that integrates both LLM and Human in the loop based on real-world doctor-patient dialogues and knowledge graphs, DISC-MedLLM boasts several features:

* **Knowledge-intensive and reliable**
* **Ability of multi-turn inquiry**
* **Alignment with human preferences**

<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/data_construction.png" alt="data-construction" width="75%"/>

## Demo
### Consultation
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/consultation.gif" alt="sample1" width="50%"/>

### Treatment Inquiry
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/advice.gif" alt="sample2" width="50%"/>

## Dataset Info
<!-- In order to align the distribution of actual doctor responses with the intended AI doctor response distribution, our dataset is constructed from five main resources: Real-world Conversations (420k), Knowledge Graph-derived Question-Answer pairs (50k), Artificially Annotated Data aligned with human preferences (2k), MedMCQA (8k), and additional general data (34k). -->

To train DISC-MedLLM, we construct a high-quality dataset called DISC-Med-SFT consisting of 480k distinct examples derived from existing medical datasets. We adopt a goal-oriented strategy by selectively reconstructing the dataset using a few deliberately chosen sources. These data sources serve the purpose of assisting LLMs in acquiring medical domain knowledge, aligning behavioral patterns with human preferences, and capturing real-world online medical dialogue distributions.

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

### Re-constructed AI doctor-patient dialogues
<!-- <img src="https://github.com/t3acup/DISC-MED/blob/main/images/figure1.png" alt="Training"/> -->

The real-world conversation data is decomposed from [MedDialog](https://github.com/UCSD-AI4H/Medical-Dialogue-System) and [cMedQA2](https://github.com/zhangsheng93/cMedQA2). Our approach employs the language ability of general LLMs to reconstruct the entire dialogue.An example of a Real-world Conversation process is as follows:

<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/adaption.png" alt="adaption" width="50%"/>

### Knowledge Graph QA pairs
We constructed some QA pairs based on [CMeKG](https://github.com/king-yyf/CMeKG_tools) with the help of properly designed prompts for the GPT-3.5 model in two steps: 

1. Transform the sampled knowledge into simple natural language QA pairs in the format (instruction, knowledge)

2. Build diverse medical scenario single-turn conversations based on these simple QA pairs.

<!-- ### Human Preferences Guided Conversation Samples -->
### Behavioral Preference Dataset
We manually selected 2,000 high-quality samples from MedDialog and cMedQA2 datasets, untouched in previous processes. After adapting some with GPT-4 and manual revisions, we used a few-shot technique to guide GPT-3.5 in generating 2,000 superior behavior-tuning samples, aligned with human preferences.

### [MedMCQA](https://github.com/medmcqa/medmcqa)
A large-scale, Multiple-Choice Question Answering (MCQA) dataset designed to address real-world medical entrance exam questions.We utilize it to generate professional medical QA samples to enhance the model's expertise in Q&A capabilities.

### General
To diversify our training set and prevent skill degradation, we incorporated general data alongside medical content during SFT training. We utilized samples from [moss-sft-003](https://huggingface.co/fnlp/moss-moon-003-sft) and [alpaca_gpt4_data_zh](https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/blob/main/data/alpaca_gpt4_data_zh.json), including 33k Brainstorming, Role Playing, and Harmless category samples from moss-sft-003, and 1k randomly chosen instances from alpaca gpt4 data zh.


### Download
You can download our dataset at <Reserved for link of dataset\>
<!-- ## Training
We underwent two stages of training, both using 4*A800 GPUs. In the first stage, we trained on a dataset comprising 420k adapted AI doctor-patient dialogues, 50k knowledge graph-based QA pairs (MedMCQA), and moss-sft-003. Hyperparameters: global batch size 24, learning rate 1e-5 (AdamW optimizer), 1 epoch, max sequence length 2048 tokens, warm-up steps 1800, no weight decay.

In the second stage, known as behavior fine-tuning, we combined a 2k carefully curated AI doctor-patient dialogue dataset with 1k alpaca gpt4 zh data. Hyperparameters: global batch size 8, learning rate 5e-6 (AdamW optimizer), 1 epoch, max sequence length 2048 tokens, no weight decay.

<img src="https://github.com/t3acup/DISC-MED/blob/main/images/figure3.png" alt="Training" width="75%"/> -->

## Deploy
The current version of DISC-MedLLM is derived from the [Baichuan-13B-Base](https://github.com/baichuan-inc/Baichuan-13B). You can directly download our model weights from the HuggingFace [repository](https://huggingface.co/Flmc/DISC-MedLLM), or automatically obtain them through the demo code.

Firstly, you need to install the requirments.
```shell
pip install -r requirements.txt
```

* Run CLI Demo
```shell
python cli_demo.py
```
* Run Web Demo
```shell
streamlit run web_demo.py --server.port 8888
```

Additionally, since the current version uses Baichuan as the base model, you can refer to its [repo](https://github.com/baichuan-inc/Baichuan-13B) for deploying with int8, int4 quantized inference. However, using quantized deployment will result in performance degradation.

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

You can see the evalution set, dialogues generated by each model and scores provided by GPT-4 in `eval` folder.<br>

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



<!-- #### Open-ended QA questions:
We selected [WebMedQA](https://github.com/hejunqing/webMedQA) and [Huatuo-26M](https://github.com/FreedomIntelligence/Huatuo-26M) as our samples. WebMedQA comprises 63,284 medical questions gathered from online health platforms. In contrast, Huatuo-26M is a vast dataset with 26 million medical QA pairs from online consultations and knowledge sources. We extracted 10% from the test sets of both datasets for our study.

#### webMedQA Evaluation Results

| Models            | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | GLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | Distinct-1 | Distinct-2 |
|-------------------|-------|-------|-------|-------|------|---------|---------|---------|------------|------------|
| Baichuan-13b-Chat | 11.15 | 3.2   | 1.18  | 0.59  | 3.22 | 14.66   | 1.43    | 10.24   | 0.73       | 0.96       |
| BianQue-2         | 12.62 | 3.97  | 1.56  | 0.82  | 3.8  | 15.94   | 1.92    | 11.48   | 0.68       | 0.94       |
| GPT-3.5           | 10.99 | 3.36  | 1.2   | 0.59  | 3.16 | 14.88   | 1.58    | 10.69   | 0.63       | 0.93       |
| HuatuoGPT(13B)    | 12.01 | 3.77  | 1.43  | 0.74  | 3.58 | 15.3    | 1.78    | 10.41   | 0.76       | 0.96       |
| DISC-MedLLM       | 11.17 | 3.32  | 1.2   | 0.61  | 3.21 | 14.69   | 1.52    | 9.39    | 0.74       | 0.97       |

#### Huatuo-26M Evaluation Results

| Models        | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | GLEU | ROUGE-1 | ROUGE-2 | ROUGE-L | Distinct-1 | Distinct-2 |
|---------------|-------|-------|-------|-------|------|---------|---------|---------|------------|------------|
| baichuan-chat | 10.26 | 3.1   | 1.18  | 0.65  | 3.03 | 14.38   | 1.56    | 10.26   | 0.75       | 0.96       |
| bianque-v2    | 12.27 | 4.02  | 1.59  | 0.86  | 3.7  | 16.31   | 2.07    | 11.89   | 0.68       | 0.93       |
| GPT-3.5-turbo | 9.72  | 3.16  | 1.24  | 0.65  | 2.87 | 14.6    | 1.76    | 10.59   | 0.64       | 0.93       |
| HuatuoGPT     | 12.08 | 3.85  | 1.48  | 0.81  | 3.58 | 15.84   | 1.95    | 10.81   | 0.76       | 0.96       |
| DISC-MedLLM   | 10.64 | 3.25  | 1.29  | 0.67  | 3.11 | 14.66   | 1.64    | 9.5     | 0.75       | 0.97       |


In open-ended QA scenarios, the landscape shifts. GPT-3.5 and our model find themselves at the lower end of the rankings, with BianQue-2 leading and Huatuo following. Yet, current open-ended QA datasets aren't ideal for assessing conversational LLMs. Reference answer language patterns often greatly contrast with those of LLMs. Consequently, lower evaluation scores may not accurately reflect the LLMs' response quality. -->

## Acknowledgement
This project wouldn't have been possible without the support and contributions of various individuals, teams, and organizations. Special thanks go to:

<!-- **MedDialog**:

**cMeKG**:

**cMedQA**:

**Baichuan-13B**:

**FireFly**: -->

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
