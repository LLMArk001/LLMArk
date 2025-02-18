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

## Usage

### Environment



## 📥 Model Download

### Basic Information
| Property     | Specification                    |
| -------- | ----------------------- |
| Model Name | LLMArk-v1.0      |
| Task Type | Multimodal decision making/prediction/question answering |
| Infrastructure | Siglip-MLP-LLM |

### Download method
```bash
# Clone this model repository
git clone https://huggingface.co/LLMArk001/LLMArk

# HuggingFace
from transformers import AutoModel
model = AutoModel.from_pretrained("LLMArk001/LLMArk")
