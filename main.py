import torch
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    GenerationConfig,
    BitsAndBytesConfig
)

from mapper import get_instruction, schema_map, split_num_mapper

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = 'model/oneke'
# autoconfig
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# 4bit量化OneKE
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    llm_int8_threshold=6.0,
    llm_int8_has_fp16_weight=False,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    device_map="auto",
    quantization_config=quantization_config,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()

task = 'NER'
language = 'zh'
schema = schema_map[task]
split_num = split_num_mapper[task]
input = "刘志坚先生：1956年出生，中国国籍，无境外居留权，中共党员，大专学历，高级经济师。"
instruction = get_instruction(language, task, schema, [], input)
sintruct = '[INST] ' + str(instruction) + '[/INST]'




input_ids = tokenizer.encode(sintruct, return_tensors="pt").to(device)
input_length = input_ids.size(1)
generation_output = model.generate(input_ids=input_ids,
                                   generation_config=GenerationConfig(max_length=1024, max_new_tokens=512,
                                                                      return_dict_in_generate=True),
                                   pad_token_id=tokenizer.eos_token_id)
generation_output = generation_output.sequences[0]
generation_output = generation_output[input_length:]
output = tokenizer.decode(generation_output, skip_special_tokens=True)

print(output)
