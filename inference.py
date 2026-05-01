import argparse
import os
import csv
import re
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Confidence threshold against FULL vocabulary
# If best digit's absolute prob in full vocab is low, model prefers non-digit tokens
CONFIDENCE_THRESHOLD = 0.03

SYSTEM_PROMPT = """You are an expert in deep learning, CNNs, and machine learning with PhD-level knowledge.

Key concepts you know well:
- CNN: convolution, pooling, stride, padding, receptive field, feature maps
- Backpropagation, vanishing/exploding gradients, chain rule
- Optimizers: SGD, Adam, RMSprop, momentum, learning rate scheduling
- Activation functions: ReLU, sigmoid, tanh, softmax, GELU
- Regularization: dropout, batch normalization, layer normalization, weight decay
- Loss functions: cross-entropy, MSE, focal loss
- Architectures: ResNet, VGG, Inception, EfficientNet, ViT, BERT, transformers
- Training: overfitting, underfitting, bias-variance tradeoff, data augmentation

Here are examples of how to reason:

Example 1:
Q: What is the output size of a 32x32 image after a Conv2D with 64 filters, 3x3 kernel, stride=1, padding=0?
Reasoning: Output = (input - kernel + 2*padding) / stride + 1 = (32 - 3 + 0) / 1 + 1 = 30. So output is 30x30x64.
Answer: 2

Example 2:
Q: Which of the following helps prevent vanishing gradients in deep networks?
A. Sigmoid activation  B. ReLU activation  C. Small learning rate  D. No skip connections
Reasoning: Sigmoid squashes gradients to near zero in deep networks. ReLU has gradient=1 for positive values, so it does not vanish. Skip connections (ResNet) also help but option B is the direct answer.
Answer: 2

Example 3:
Q: In batch normalization, what is normalized?
A. Weights  B. Gradients  C. Activations  D. Loss
Reasoning: Batch norm normalizes the activations (outputs) of each layer across the batch to have zero mean and unit variance.
Answer: 3

Now look at the MCQ image carefully and answer the question shown.

Step 1: Read the question and all 4 options carefully.
Step 2: Apply your deep learning knowledge to reason through each option.
Step 3: Eliminate wrong options and select the best answer.

End your response with exactly: "Answer: X" where X is 1, 2, 3, or 4.
1 = Option A
2 = Option B
3 = Option C
4 = Option D

If the image is unreadable or the question makes no sense, do NOT write "Answer:" at all."""

def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_7b = os.path.join(script_dir, "Qwen2-VL-7B-Instruct")
    local_2b = os.path.join(script_dir, "Qwen2-VL-2B-Instruct")

    if os.path.exists(local_7b):
        model_path = local_7b
    elif os.path.exists(local_2b):
        model_path = local_2b
    else:
        # No local weights — will download at runtime (needs internet)
        model_path = "Qwen/Qwen2-VL-7B-Instruct"

    print(f"Loading model from: {model_path}")

    # Detect available GPUs and split model memory accordingly
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        # Split across two GPUs (e.g. Kaggle T4 x2: 2x15GB)
        max_memory = {i: "13GiB" for i in range(n_gpus)}
        max_memory["cpu"] = "20GiB"
    elif n_gpus == 1:
        max_memory = {0: "13GiB", "cpu": "20GiB"}
    else:
        max_memory = None

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
    )
    processor = AutoProcessor.from_pretrained(model_path)
    return model, processor

def get_logprob_answer(inputs, model, processor):
    """Chain-of-thought greedy pass, then check digit probability against FULL vocabulary."""
    digit_token_ids = [
        processor.tokenizer.encode(str(d), add_special_tokens=False)[0]
        for d in range(1, 5)
    ]

    with torch.no_grad():
        cot_output = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=None,
            top_p=None,
        )

    cot_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, cot_output)
    ]
    cot_text = processor.batch_decode(cot_trimmed, skip_special_tokens=True)[0].strip()

    # Append "Answer:" and check what the model wants to generate next
    answer_suffix = processor.tokenizer.encode("\nAnswer:", add_special_tokens=False, return_tensors="pt")[0].to(inputs.input_ids.device)
    full_ids = torch.cat([cot_output[0], answer_suffix]).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_ids=full_ids).logits[0, -1]  # shape: [vocab_size]

    # Key fix: use FULL vocabulary softmax, not just over 4 digits
    # This way if the model prefers non-digit tokens, digit probs will be low
    full_probs = F.softmax(logits, dim=0)
    digit_probs = [full_probs[tid].item() for tid in digit_token_ids]

    best_idx = int(torch.tensor(digit_probs).argmax().item())
    best_prob = digit_probs[best_idx]
    best_digit = best_idx + 1

    return cot_text, best_digit, best_prob


def predict_answer(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    # Cap image size to limit vision encoder patch count (avoids OOM on 15GB GPUs)
    max_side = 800
    if max(image.size) > max_side:
        ratio = max_side / max(image.size)
        image = image.resize((int(image.size[0] * ratio), int(image.size[1] * ratio)), Image.LANCZOS)

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

    # Greedy chain-of-thought pass + logprob confidence check
    cot_text, answer, confidence = get_logprob_answer(inputs, model, processor)
    print(f"  Reasoning: {cot_text[:120]}...")
    print(f"  Answer: {answer} | Confidence: {confidence:.2f}")

    if confidence < CONFIDENCE_THRESHOLD:
        print(f"  -> Low confidence, skipping (output 5)")
        return 5

    return answer


def main():
    import time

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
    total_start = time.time()

    for idx, image_name in enumerate(image_names):
        image_path = os.path.join(images_dir, f"{image_name}.png")
        if not os.path.exists(image_path):
            print(f"Warning: {image_path} not found, skipping (marking as 5)")
            results.append({"image_name": image_name, "option": 5})
            continue

        img_start = time.time()
        print(f"Processing {image_name} ({idx+1}/{len(image_names)})...")
        answer = predict_answer(image_path, model, processor)
        elapsed = time.time() - img_start
        total_elapsed = time.time() - total_start
        remaining = (total_elapsed / (idx + 1)) * (len(image_names) - idx - 1)
        print(f"  -> Answer: {answer} | {elapsed:.1f}s | ETA: {remaining/60:.1f}min")
        results.append({"image_name": image_name, "option": answer})

    # Write submission.csv in the script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    submission_path = os.path.join(script_dir, "submission.csv")
    with open(submission_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image_name", "option"])
        writer.writeheader()
        writer.writerows(results)

    total_time = time.time() - total_start
    print(f"\nSubmission saved to {submission_path}")
    print(f"Total time: {total_time/60:.1f} min ({total_time:.0f}s)")


if __name__ == "__main__":
    main()
