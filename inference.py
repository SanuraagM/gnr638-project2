import argparse
import os
import csv
import re
import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

SYSTEM_PROMPT = """You are an expert in deep learning and machine learning.
You will be shown an image containing a multiple choice question about deep learning.
Read the question and all options carefully, then select the correct answer.
Respond with ONLY a single digit: 1, 2, 3, or 4 corresponding to option A, B, C, D respectively.
If you are not sure at all, respond with 5.
Do not explain. Do not write anything else. Just one digit."""

def load_model():
    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_name)
    return model, processor

def predict_answer(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": SYSTEM_PROMPT},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()

    # Extract first digit from output
    match = re.search(r"[1-5]", output_text)
    if match:
        return int(match.group())
    return 5  # unanswered if no valid digit found


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test directory")
    args = parser.parse_args()

    test_dir = args.test_dir
    test_csv_path = os.path.join(test_dir, "test.csv")
    images_dir = os.path.join(test_dir, "images")

    # Read test.csv
    image_names = []
    with open(test_csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            image_names.append(row["image_name"])

    print(f"Found {len(image_names)} images to process")

    # Load model
    print("Loading Qwen2-VL-7B model...")
    model, processor = load_model()
    print("Model loaded successfully")

    # Run inference
    results = []
    for image_name in image_names:
        image_path = os.path.join(images_dir, f"{image_name}.png")
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping (marking as 5)")
            results.append({"image_name": image_name, "option": 5})
            continue

        print(f"Processing {image_name}...")
        answer = predict_answer(image_path, model, processor)
        print(f"  -> Answer: {answer}")
        results.append({"image_name": image_name, "option": answer})

    # Write submission.csv in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_path = os.path.join(script_dir, "submission.csv")
    with open(submission_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "option"])
        writer.writeheader()
        writer.writerows(results)

    print(f"\nSubmission saved to {submission_path}")


if __name__ == "__main__":
    main()
