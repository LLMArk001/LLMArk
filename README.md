# 🌊 LLMArk: Multimodal Large Language Model For Flood Risk Assessment

**LLMArk: Multimodal Large Language Model For Flood Risk Assessment**  
![License](https://img.shields.io/badge/License-Apache2.0-blue)
![Version](https://img.shields.io/badge/Version-1.0.0-red)
![Params](https://img.shields.io/badge/Parameters-8B-yellowgreen)

🚨 **Real-time rescue decision-making** | 📡 **Multi-modal perception** | 🧭 **Geospatial Understanding**  

## Introduction

In this study, the proposed LLMArk utilizes vision and language modalities to enhance the perception, understanding and reasoning in flood scenarios, which integrates the visual linguistic processing capabilities of MLLMs with flood domain-specific knowledge as well as large-scale vision-language dataset construction. 
The main contributions of this study is to propose a unified expert of flood-affected object detector, flood risk assessor, and rescue guidance generator, allowing terminals to interactively inquire flood risk severity and request rescue guidance. 
By analyzing extensive experimental results, LLMArk demonstrates its outstanding performance in supporting multi-round multimodal dialogues, offering accurate perception of object locations, delivering reliable understanding of risk levels, and conducting rational reasoning for generating rescue guidance. 

## Installation

```
git clone https://github.com/LLMArk001/LLMArk.git
cd LLMArk
conda create -n llmark python=3.10
conda activate llmark
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118
pip install .
```

## 📥 Download our trained weights

### Basic Information 
| Property     | Specification                    |
| -------- | ----------------------- |
| Model Name | LLMArk-v1.0      |
| Task Type | Multimodal decision making/prediction/question answering |

**Our previous trained weights for Flood Risk Assessment could be downloaded at [LLMArk weights](https://huggingface.co/LLMArk001/LLMArk).**

**The instance perception weights used by our model can be obtained from [Instance_perception.pt](https://huggingface.co/LLMArk001/LLMArk/resolve/main/Instance_perception.pt).**

## Code of Inference

```python
from llmark import LLMArkQwenForCausalLM
from llmark.model.builder import load_pretrained_model
from llmark.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llmark.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llmark.conversation import conv_templates, SeparatorStyle
from llmark.get_box import prompt_boxes
import copy
import torch
from PIL import Image
import warnings

warnings.filterwarnings("ignore")
pretrained = "LLMArk001/LLMArk"
device = "cuda"
device_map = "auto"
tokenizer, model, image_processor, max_length = load_pretrained_model(pretrained, device_map=device_map)
model.eval()

image_path = "global_imgs/Vietnam2.jpg"
question = "What is the overall flooding risk in this scene, and what general recommendations can be given?"
image = Image.open(image_path)
image_tensor = process_images([image], image_processor, model.config)
image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]
question = DEFAULT_IMAGE_TOKEN + "\n" + question
conv = copy.deepcopy(conv_templates)
conv.append_message(conv.roles[0], question)
conv.append_message(conv.roles[1], None)
prompt_question = conv.get_prompt()
prompt_question = prompt_boxes(prompt_question, image_path)
input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)
image_sizes = [image.size]

cont = model.generate(
	input_ids,
	images=image_tensor,
	image_sizes=image_sizes,
	do_sample=False,
	temperature=1,
	top_k=50,
	top_p=0.95,
	max_new_tokens=4096,
)

text_outputs = tokenizer.batch_decode(cont, skip_special_tokens=True)
print(text_outputs[0])
```
