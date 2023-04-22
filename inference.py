"""
Donut
Copyright (c) 2022-present NAVER Corp.
MIT License
"""
import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from PIL import Image
from tqdm import tqdm

from donut import DonutModel, JSONParseEvaluator, load_json, save_json


def test(args):
    pretrained_model = DonutModel.from_pretrained(args.pretrained_model_name_or_path)
    image = Image.open(args.img_path).convert("RGB")

    if torch.cuda.is_available():
        pretrained_model.half()
        pretrained_model.to("cuda")

    pretrained_model.eval()

    if args.save_path:
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)

    predictions = []
    output = pretrained_model.inference(image=image, prompt=f"<s_{args.task_name}>")["predictions"][0]

    predictions.append(output)

    #if args.save_path:
    with open("pred_result.json", "w") as f:
        json.dump(predictions[0], f, indent=4)

    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_name_or_path", type=str)
    parser.add_argument("--task_name", type=str, default=None)
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--img_path", type=str)
    args, left_argv = parser.parse_known_args()
    args.task_name = "example"

    predictions = test(args)
    print(predictions)