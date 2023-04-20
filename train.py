from dataset_ import get_data
from donut_preprocessor import DonutDataset

TRAIN_DIR = "dataset/training_data"
TEST_DIR = "dataset/testing_data"

train_dataset = get_data(TRAIN_DIR)
#test_dataset = get_data(TEST_DIR)

#example = train_dataset[0]['gt_parse']

#print(example)

donut_train_dataset = DonutDataset(dataset = train_dataset, split="train", max_length=768, 
                                   task_start_token="<funsd>", prompt_end_token="<funsd>",
                                   sort_json_key=False)

pixel_values, labels, target_sequence = donut_train_dataset[0]

print(pixel_values)
print("=======================")
print(labels)
print("=======================")
print(target_sequence)

# let's print the labels (the first 30 token ID's)
for id in labels.tolist()[:30]:
  if id != -100:
    print(donut_train_dataset.processor.decode([id]))
  else:
    print(id)