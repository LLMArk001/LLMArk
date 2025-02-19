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
pip install -e ".[train]"
```

## 📥 Download our trained weights

### Basic Information 
| Property     | Specification                    |
| -------- | ----------------------- |
| Model Name | LLMArk-v1.0      |
| Task Type | Multimodal decision making/prediction/question answering |

**Our previous trained weights for Flood Risk Assessment could be downloaded at [LLMArk weights](https://huggingface.co/LLMArk001/LLMArk).**

**The instance perception weights used by our model can be obtained from [Instance_perception.pt](https://huggingface.co/LLMArk001/LLMArk/resolve/main/Instance_perception.pt).**

## Launching Demo Locally

The demo code **demo.py** for the example we provided allows you to change the paths to the images and weights in the code to achieve local operation.
```bash
python demo.py
```