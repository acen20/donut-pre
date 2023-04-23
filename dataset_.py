from utils import normalize_bbox, load_image
import json
import os
from datasets import Dataset
from donut_preprocessor import DonutDataset

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
            other_labels = {}
            file_path = os.path.join(ann_dir, file)
            with open(file_path, "r", encoding="utf8") as f:
                data = json.load(f)
            image_path = os.path.join(img_dir, file)
            image_path = image_path.replace("json", "png")
            image, size = load_image(image_path)
            for item in data["form"]:
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue
                    
                if label not in ["answer", "question"]:
                    obj = {"text_sequence":item["text"]}
                    if label not in other_labels.keys():
                        other_labels[label] = [obj]
                    else:
                        other_labels[label].append(obj)
                
                if "question" in label.lower():
                    obj = {}
                    question = item["text"]
                    answer = ""
                    if item["linking"]:
                        answer = [a for a in data["form"] if a["id"] == item["linking"][0][1]]
                        answer = answer[0]["text"] if answer else ""
                    obj["question"] = question
                    obj["answer"] = answer
                    pairs.append(obj)  

            gt_parse = {
                "pairs":pairs
            }

            gt_parse.update(other_labels)

            yield {"image":image, "ground_truth":{"gt_parse": gt_parse}}


def get_data(filepath):
    custom_data = CustomDataset()
    data = Dataset.from_generator(custom_data._generate_examples,
                                gen_kwargs={'filepath':f'{filepath}'})
    '''with open("metadata.jsonl", "w") as f:
        for sample in data:                             
            f.write(json.dumps(sample))
            f.write("\n")'''
    
    return data

if __name__ == "__main__":
    TRAIN_DIR = "dataset/training_data"
    VAL_DIR = "dataset/testing_data"

    train_dataset = get_data(TRAIN_DIR)
    val_dataset = get_data(VAL_DIR)

    donut_train_dataset = DonutDataset(dataset = train_dataset, split="train", max_length=768, 
                                   task_start_token="<s_custom>", prompt_end_token="<s_custom>",
                                   sort_json_key=False)
    
    print(donut_train_dataset[0])