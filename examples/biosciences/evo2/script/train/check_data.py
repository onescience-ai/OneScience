#!/usr/bin/env python3

import os
import struct



def main():
    data_dir = "/public/home/onescience2025404/khren/onescience/data/promoters/pretraining_data_promoters"
    # 检查所有.bin文件
    bin_files = [
        "data_promoters_train_text_CharLevelTokenizer_document.bin",
        "data_promoters_valid_text_CharLevelTokenizer_document.bin", 
        "data_promoters_test_text_CharLevelTokenizer_document.bin"
    ]    

    
    train_file = os.path.join(data_dir, "data_promoters_train_text_CharLevelTokenizer_document.bin")

 

if __name__ == "__main__":
    main()
