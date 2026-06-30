# detect_image_token.py
import os
import sys
import argparse
from transformers import AutoProcessor
from PIL import Image

def detect_image_token():
    parser = argparse.ArgumentParser(description='Detect correct image token for MedGemma model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to local MedGemma model directory')
    args = parser.parse_args()

    model_path = args.model_path
    
    print(f"Loading processor from: {model_path}")
    
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True, use_fast=False)
        tokenizer = processor.tokenizer
        
        print("Inspecting tokenizer for image-related tokens...")
        
        # Check various attributes that might contain image token info
        attrs_to_check = [
            'image_token', 'image_token_id',
            'pad_token', 'bos_token', 'eos_token', 'unk_token',
            'additional_special_tokens', 'additional_special_tokens_ids'
        ]
        
        for attr in attrs_to_check:
            try:
                value = getattr(tokenizer, attr, None)
                if value is not None:
                    print(f"{attr}: {value}")
            except:
                print(f"{attr}: Could not access")
        
        # Look for common image tokens in vocab
        common_image_tokens = [
            "<image>", "<img>", "<IMG>", "<IMAGE>", 
            "<|image|>", "[IMG]", "[IMAGE]",
            "<start_of_image>", "<end_of_image>",
            "<image_soft_token>", "<image_token>",
            "<img>", "</img>", "<IMG>", "</IMG>",
            "image", "IMAGE", "img", "IMG",
            "<vision>", "</vision>", "<visual>", "</visual>"
        ]
        
        print("\nChecking common image tokens:")
        for token in common_image_tokens:
            try:
                token_id = tokenizer.convert_tokens_to_ids(token)
                # If the token exists in vocab, convert_tokens_to_ids returns its ID
                # Otherwise it returns the unk_token_id
                unk_id = tokenizer.unk_token_id
                if token_id != unk_id:
                    print(f"Found token '{token}' with ID: {token_id}")
                else:
                    print(f"Token '{token}' not in vocab (would use UNK token ID: {unk_id})")
            except Exception as e:
                print(f"Error checking token '{token}': {e}")
        
        # Also test some sample prompts with different image tokens
        print("\nTesting sample prompts with different image tokens:")
        test_prompts = [
            "Describe this image: <image>",
            "What do you see? <image>",
            "Analyze: <image>",
            "Look at this: <image>",
            "Explain: <|image|>",
            "Look: <start_of_image><image_soft_token><end_of_image>"
        ]
        
        for i, prompt in enumerate(test_prompts[:3]):  # Only test first few to avoid errors
            try:
                print(f"\nTest {i+1}: '{prompt}'")
                inputs = processor(text=prompt, return_tensors="pt")
                print(f"  Input IDs shape: {inputs['input_ids'].shape}")
                # Decode to see actual tokens
                decoded = tokenizer.decode(inputs['input_ids'][0])
                print(f"  Decoded: {decoded[:200]}...")
            except Exception as e:
                print(f"  Error: {e}")
        
    except Exception as e:
        print(f"Error loading processor: {e}")

if __name__ == "__main__":
    detect_image_token()
