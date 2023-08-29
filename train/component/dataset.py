import json
from loguru import logger
from torch.utils.data import Dataset


class SFTDataset(Dataset):
    def __init__(self, file, tokenizer, max_seq_length):
        self.tokenizer = tokenizer
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.user_token_id = 195
        self.assistant_token_id = 196
        self.eos_token = tokenizer.eos_token
        self.bos_token = tokenizer.bos_token
        self.max_seq_length = max_seq_length
        logger.info('Loading data: {}'.format(file))
        with open(file, 'r', encoding='utf8') as f:
            data_list = f.readlines()
        logger.info("there are {} data in dataset".format(len(data_list)))
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    # def __getitem__(self, index):
    #     # 每条数据格式为: <s><user>input1<assi>target1</s><user>input2<assi>target2</s>...
    #     data = self.data_list[index]
    #     data = json.loads(data)
    #     conversation = data['conversation']

    #     # 收集多轮对话
    #     utterances = []
    #     for i, message in enumerate(conversation[::-1]):
    #         utterances.append(x['human'])
    #         utterances.append(x['assistant'])
    #     utterances_ids = self.tokenizer(utterances, add_special_tokens=False).input_ids

    #     # 模型的输入格式为：<s>input1</s>target1</s>input2</s>target2</s>...
    #     input_ids = [self.bos_token_id]
    #     target_mask = [0]  # 用于对input进行mask，只计算target部分的loss
    #     for i, utterances_id in enumerate(utterances_ids):
    #         input_ids += (utterances_id + [self.eos_token_id])
    #         if i % 2 == 0:
    #             target_mask += [0] * (len(utterances_id) + 1)
    #         else:
    #             target_mask += [1] * (len(utterances_id) + 1)
    #     assert len(input_ids) == len(target_mask)
    #     # 对长度进行截断
    #     input_ids = input_ids[:self.max_seq_length]
    #     target_mask = target_mask[:self.max_seq_length]
    #     attention_mask = [1] * len(input_ids)
    #     assert len(input_ids) == len(target_mask) == len(attention_mask)
    #     inputs = {
    #         'input_ids': input_ids,
    #         'attention_mask': attention_mask,
    #         'target_mask': target_mask
    #     }
    #     return inputs

    def __getitem__(self, index):
        # 每条数据格式为: <s>input1</s>target1</s>input2</s>target2</s>...
        data = self.data_list[index]
        data = json.loads(data)
        conversation = data['conversation']

        max_seq_length = self.max_seq_length
        # max_input_tokens = self.config.model_max_length - max_new_tokens
        # max_input_tokens = max(self.config.model_max_length // 2, max_input_tokens)
        total_input, target_mask = [self.bos_token_id], [0]
        for i, message in enumerate(conversation):
            round_input = []
            content_tokens = self.tokenizer.encode(message['content'])
            if message['role'] == 'user':
                round_input = [self.user_token_id] + content_tokens
                if total_input and len(total_input) + len(round_input) > max_seq_length:
                    break
                else:
                    target_mask += [0]*len(round_input)
                    total_input += round_input
            elif message['role'] == 'assistant':
                round_input = [
                    self.assistant_token_id
                ] + content_tokens + [
                    self.eos_token_id
                ]
                if total_input and len(total_input) + len(round_input) > max_seq_length:
                    break
                else:
                    total_input += round_input
                    target_mask += [0]
                    target_mask += [1]*(len(round_input)-1)

        input_ids = total_input[:self.max_seq_length]
        target_mask = target_mask[:self.max_seq_length]
        attention_mask = [1] * len(input_ids)
        assert len(input_ids) == len(target_mask) == len(attention_mask)
        
        inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'target_mask': target_mask
        }
        return inputs