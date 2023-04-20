from utils import normalize_bbox, load_image
import json
import os
from datasets import Dataset

class CustomDataset():

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, filepath):
        ann_dir = os.path.join(filepath, "annotations")
        img_dir = os.path.join(filepath, "images")
        for guid, file in enumerate(sorted(os.listdir(ann_dir))):
            pairs = []
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["form"]:
                if item["label"] in ["header", "other"]:
                    continue
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                
                if label == "question":
                    question = item["text"]
                    answer = ""
                    if item["linking"]:
                        answer = [a for a in data["form"] if a["id"] == item["linking"][0][1]]
                        answer = answer[0]["text"] if answer else ""

                    pairs.append({"question":question, "answer": answer})

        
                
            yield {"gt_parse": {"pairs":pairs}, "image":image}


def get_data(filepath):
    custom_data = CustomDataset()
    data = Dataset.from_generator(custom_data._generate_examples,
                                gen_kwargs={'filepath':f'{filepath}'})
                             

    return data
