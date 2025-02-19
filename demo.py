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

image_path = "path/to/your/image"
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