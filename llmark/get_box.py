from ultralytics import YOLO
import re
from PIL import Image

def get_boxes(string):
	coordinates = re.findall(r'\[\d+\.\d+,\s?\d+\.\d+,\s?\d+\.\d+,\s?\d+\.\d+\]', string)
	return coordinates

def get_boxes_from_image(image_path):
    image = Image.open(image_path)

    w,h = image.size
    yolo_model = YOLO("LLMArk001/LLMArk/instance.pt")  # pretrained YOLO11n model

    results = yolo_model(image_path, conf=0.4)
    results[0].save(filename="result.jpg", conf=False, probs=False, labels=False)
    all_boxes = []
    boxes = results[0].boxes.data.tolist()
    for box in boxes:
        x1 = box[0]
        y1 = box[1]
        x2 = box[2]
        y2 = box[3]

        zb = [round(x1/w, 3), round(y1/h, 3), round(x2/w, 3), round(y2/h, 3)]
        all_boxes.append(zb)
    return all_boxes

def prompt_boxes(prompt_question, image_path):
    all_boxes = get_boxes_from_image(image_path)
    prompt_boxes = get_boxes(prompt_question)     
    if len(prompt_boxes) == 0:
        prompt_question = prompt_question.replace(" in ",f" at position {str(all_boxes)} in ")
    return prompt_question