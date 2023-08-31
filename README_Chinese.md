# DISC-MedLLM

<div align="center">
  
[![Generic badge](https://img.shields.io/badge/🤗-Huggingface%20Repo-green.svg)](https://huggingface.co/Flmc/DISC-MedLLM)
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/FudanDISC/DICS-MedLLM/blob/main/LICENSE)
<br>
</div>
<div align="center">

[Demo](http://medllm.fudan-disc.com) | [技术报告](https://arxiv.org/abs/2308.14346)
<br>
中文 | [EN](https://github.com/FudanDISC/DISC-MedLLM/blob/main/README_EN.md)
</div>
  
DISC-MedLLM 是一个专门针对医疗健康对话式场景而设计的医疗领域大模型，由[复旦大学数据智能与社会计算实验室 (Fudan-DISC)](http://fudan-disc.com) 开发并开源。

该项目包含下列开源资源:
* [DISC-Med-SFT 数据集](https://huggingface.co/datasets/Flmc/DISC-Med-SFT) (不包括行为偏好训练数据)
* DISC-MedLLM 的[模型权重](https://huggingface.co/Flmc/DISC-MedLLM)

您可以通过访问这个[链接](http://medllm.fudan-disc.com)来试用我们的模型。

## 概述

DISC-MedLLM 是一个专为医疗健康对话场景而打造的领域大模型，它可以满足您的各种医疗保健需求，包括疾病问诊和治疗方案咨询等，为您提供高质量的健康支持服务。

DISC-MedLLM 有效地对齐了医疗场景下的人类偏好，弥合了通用语言模型输出与真实世界医疗对话之间的差距，这一点在实验结果中有所体现。

得益于我们以目标为导向的策略，以及基于真实医患对话数据和知识图谱，引入LLM in the loop 和 Human in the loop的多元数据构造机制，DISC-MedLLM 有以下几个特点：

* **可靠丰富的专业知识**，我们以医学知识图谱作为信息源，通过采样三元组，并使用通用大模型的语言能力进行对话样本的构造。
* **多轮对话的问询能力**，我们以真实咨询对话纪录作为信息源，使用大模型进行对话重建，构建过程中要求模型完全对齐对话中的医学信息。
* **对齐人类偏好的回复**，病人希望在咨询的过程中获得更丰富的支撑信息和背景知识，但人类医生的回答往往简练；我们通过人工筛选，构建符合人类偏好的高质量的小规模行为微调样本，对齐病人的需求。

<数据构造图>
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/data_construction.png" alt="data-construction" width="85%"/>

## 模型效果演示
### 疾病问诊
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/consultation.gif" alt="sample1" width="60%"/>

### 治疗方案咨询
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/advice.gif" alt="sample2" width="60%"/>

## 数据集

为了训练 DISC-MedLLM ，我们构建了一个高质量的数据集，命名为 DISC-Med-SFT，其中包含了超过47万个从现有的医疗数据集中蒸馏得到的样本。我们采用了目标导向的策略，通过对于精心选择的几个数据源进行重构来得到指令微调数据集。这些数据的作用在于帮助模型学习医疗领域知识，将行为模式与人类偏好对齐，并对齐真实世界在线医疗对话的分布情况。

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
    <th class="tg-9wq8" rowspan="2"><br>数据集</th>
    <th class="tg-9wq8" rowspan="2"><br>数据来源</th>
    <th class="tg-9wq8" rowspan="2"><br>样本量</th>
  </tr>
  <tr>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-9wq8" rowspan="2">利用AI重构的医患对话</td>
    <td class="tg-9wq8">MedDialog</td>
    <td class="tg-9wq8">400k</td>
  </tr>
  <tr>
    <td class="tg-9wq8">cMedQA2</td>
    <td class="tg-c3ow">20k</td>
  </tr>
  <tr>
    <td class="tg-c3ow">知识图谱问答对</td>
    <td class="tg-9wq8">CMeKG</td>
    <td class="tg-9wq8">50k</td>
  </tr>
  <tr>
    <td class="tg-c3ow">行为偏好数据集</td>
    <td class="tg-9wq8">人为筛选</td>
    <td class="tg-9wq8">2k</td>
  </tr>
  <tr>
    <td class="tg-9wq8" rowspan="3">其他</td>
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


<br>


### 下载

我们总共发布了近47万条训练数据，其中包括重新构建的对话和知识图谱问答对。您可以访问这个[链接](https://huggingface.co/datasets/Flmc/DISC-Med-SFT)下载数据集。

<br>


## 部署

当前版本的 DISC-MedLLM 是基于[Baichuan-13B-Base](https://github.com/baichuan-inc/Baichuan-13B)训练得到的。您可以直接从 [Hugging Face](https://huggingface.co/Flmc/DISC-MedLLM) 上下载我们的模型权重，或者根据下列代码样例中的方式自动获取。

首先，您需要安装项目的依赖环境。
```shell
pip install -r requirements.txt
```

### 利用Hugging Face的transformers模块来进行推理
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

### 运行命令行Demo
```shell
python cli_demo.py
```
### 运行网页版Demo
```shell
streamlit run web_demo.py --server.port 8888
```

此外，由于目前版本的 DISC-MedLLM 是以 Baichuan-13B 作为基座的，您可以参考 [Baichuan-13B 项目](https://github.com/baichuan-inc/Baichuan-13B)的介绍来进行 int8 或 int4 量化推理部署。然而需要注意的是，使用模型量化可能会导致性能的下降。
<br>

## 对模型进行微调
您可以使用与我们的数据集结构相同的数据对我们的模型进行微调。我们的训练代码在 [Firefly](https://github.com/yangjianxin1/Firefly) 的基础上进行了修改，使用了不同的数据结构和对话格式。这里我们只提供全参数微调的代码：
```shell
deepspeed --num_gpus={num_gpus} ./train/train.py --train_args_file ./train/train_args/sft.json
```
> 请在您在开始进行模型训练前检查 `sft.json` 中的设置。

<br>如果您想使用其他训练代码来微调我们的模型，请使用如下对话格式。
```shell
<\b><$user_token>content<$assistant_token>content<\s><$user_token>content ...
```
我们使用的 `user_token` 和 `assistant_token` 分别为 `195` and `196`，这和 Baichuan-13B-Chat 是相同的。

## 模型评测
<!-- We compare our model with three general-purpose LLMs and two conversational Chinese medical domain LLMs. Specifically, these are GPT-3.5 and GPT-4 from OpenAI, the aligned conversational version of our backbone model Baichuan-13B-Base, Baichuan-13B-Chat, and the open-source Chinese conversational medical model HuatuoGPT-13B (trained from Ziya-Llama-13B) and BianQue-2. Our evaluation approach encompasses two key dimensions: an assessment of conversational aptitude using GPT-4 as a reference judge, and a comprehensive benchmark evaluation. -->

我们从两个角度评估了模型的性能，包括检测其在单轮对话中提供准确答案的能力以及在多轮对话中进行系统性问诊的能力。

* 在单轮对话评测中，我们构建了一个基准测试数据集，其中包含从两个公开医疗数据集中收集的多项选择题，并评估模型回答的准确性。
* 对于多轮对话评测，我们首先构建了一些高质量的诊疗对话案例，然后让 GPT-3.5 扮演这些案例中的患者角色，并与扮演医生角色的模型进行对话。我们利用 GPT-4 来评估了模型的**主动性**、**准确性**, **帮助性**和**语言能力**。

您可以在 `eval/` 目录下查看测试数据集、各个模型生成的对话结果以及 GPT-4 提供的打分结果。<br>

### 单轮对话评测
我们在评测中选用了 [MLEC-QA](https://github.com/Judenpech/MLEC-QA) 和考研306（西医综合）的单项选择题。
<!-- The MLEC-QA contains questions from the China NMLEC, categorized into Clinic, Stomatology, Public Health, Traditional Chinese Medicine, and Integrated Traditional Chinese and Western Medicine. We selected 1,362 questions (10% of the test set) for evaluation. From Western Medicine 306, we used a combined 270 questions from 2020 and 2021. Our study involved both zero-shot and few-shot approaches, with examples from MLEC-QA's validation set and 2019 Western Medicine 306 questions for the few-shot samples. -->

#### Few-shot  

| 模型             | MLEC-QA 临床 | MLEC-QA 中西医结合 | MLEC-QA 公共卫生 | MLEC-QA 口腔 | MLEC-QA 中医 | 考研306西医综合 | 平均 |
|-------------------|----------------|-------------|----------------------|---------------------|------------|----------|---------|
| GPT-3.5           | 58.63          | 45.9        | 53.51                | 51.52               | 43.47      | 44.81    | 49.64   |
| Baichuan-13b-Chat| 31.25          | 37.69       | 28.65                | 27.27               | 29.77      | 24.81    | 29.91   |
| Huatuo(13B)        | 31.85          | 25          | 32.43                | 32.95               | 26.54      | 24.44    | 28.87   |
| DISC-MedLLM        | 44.64          | 41.42       | 41.62                | 38.26               | 39.48      | 33.33    | 39.79   |

#### Zero-shot

| 模型             | MLEC-QA 临床 | MLEC-QA 中西医结合 | MLEC-QA 公共卫生 | MLEC-QA 口腔 | MLEC-QA 中医 | 考研306西医综合 | 平均 |
|-------------------|----------------|-------------|----------------------|---------------------|------------|----------|---------|
| GPT-3.5           | 47.32          | 33.96       | 48.11                | 39.77               | 38.83      | 33.33    | 40.22   |
| Baichuan-13b-Chat| 44.05          | 43.28       | 39.92                | 31.06               | 41.42      | 32.22    | 38.66   |
| Huatuo(13B)        | 27.38          | 21.64       | 25.95                | 25.76               | 24.92      | 20.37    | 24.34   |
| DISC-MedLLM        | 44.64          | 37.31       | 35.68                | 34.85               | 41.75      | 31.11    | 37.56   |

<!-- GPT-3.5 clearly outperformed others in the multiple-choice assessment, while our model achieved a strong second place in few-shot scenarios. In zero-shot scenarios, it followed closely behind Baichuan-13B-Chat, securing the third spot. These results highlight the current priority gap in performance for conversational medical models on knowledge-intensive tests like multiple-choice questions. -->

### 多轮对话能力评测
我们的评测基于三个不同的数据集：Chinese Medical Benchmark ([CMB-Clin](https://github.com/FreedomIntelligence/CMB))、Chinese Medical Dialogue Dataset ([CMD](https://github.com/UCSD-AI4H/Medical-Dialogue-System)) 和 Chinese Medical Intent Dataset ([CMID](https://github.com/IMU-MachineLearningSXD/CMID))，其中 CMB-Clin 模拟了现实世界的问诊过程，而 CMD 和 CMID 则分别着重从科室专业性和用户意图的角度进行评估。 <br>

<!-- Within this framework, The Evaluation of the dialogues is based on four criteria: Proactivity, Accuracy, Helpfulness, and Linguistic Quality.

1. Proactivity: The doctor can proactively and clearly request the patient to provide more information about the symptoms, physical examination results, and medical history when the information is insufficient, actively guiding the patient through the consultation process. 
2. Accuracy: The diagnosis or advice the doctor provides is accurate and has no factual errors. Conclusions are not made arbitrarily.
3. Helpfulness: The doctor's responses provide the patient with clear, instructive, and practical assistance, specifically addressing the patient's concerns.
4. Linguistic Quality: The conversation is logical. The doctor correctly understands the patient's semantics, and the expression is smooth and natural. -->

#### CMB-clin数据集的评测结果:
| **模型**              | **主动性** | **准确性** | **帮助性** | **语言能力** | **平均** |
|------------------------|-----------------|--------------|-----------------|------------------------|-------------|
| **GPT3.5**             | 4.30            | 4.53         | 4.55            | 5.00                   | 4.60        |
| **GPT4**               | 4.15            | 4.70         | 4.75            | 4.96                   | 4.64        |
| **Baichuan-13b-Caht**  | 4.30            | 4.58         | 4.73            | 4.95                   | 4.64        |
| **BianQue-2**          | 3.97            | 4.36         | 4.37            | 4.81                   | 4.38        |
| **Huatuo(13B)**        | 4.40            | 4.62         | 4.74            | 4.96                   | 4.68        |
| **DISC-MedLLM**        | 4.64            | 4.47         | 4.66            | 4.99                   | 4.69        |

#### CMD数据集的评测结果
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/cmd.png" alt="cmd" width="75%"/>

#### CMID数据集的评测结果
<img src="https://github.com/FudanDISC/DISC-MedLLM/blob/main/images/cmid.png" alt="cmid" width="75%"/>



## 致谢
本项目基于如下开源项目展开，在此对相关项目和开发人员表示诚挚的感谢：

- [**MedDialog**](https://github.com/UCSD-AI4H/Medical-Dialogue-System)

- [**cMeKG**](https://github.com/king-yyf/CMeKG_tools)

- [**cMedQA**](https://github.com/zhangsheng93/cMedQA2)

- [**Baichuan-13B**](https://github.com/baichuan-inc/Baichuan-13B)

- [**FireFly**](https://github.com/yangjianxin1/Firefly)

同样感谢其他限于篇幅未能列举的为本项目提供了重要帮助的工作。

## 声明
由于语言模型固有的局限性，我们无法保证 DISC-MedLLM 模型所生成的信息的准确性或可靠性。该模型仅为个人和学术团体的研究和测试而设计。我们敦促用户以批判性的眼光对模型输出的任何信息或医疗建议进行评估，并且强烈建议不要盲目信任此类信息结果。我们不对因使用该模型所引发的任何问题、风险或不良后果承担责任。

## 引用
如果我们的工作有帮助到您的研究，请引用我们：
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
