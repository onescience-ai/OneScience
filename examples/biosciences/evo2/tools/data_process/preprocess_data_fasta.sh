#!/bin/bash

set -e  

PROJECT_ROOT=$(python -c "from pathlib import Path; print(Path(__name__).resolve().parents[3])")
echo "ONESCIENCE_PATH:" $PROJECT_ROOT

cd $PROJECT_ROOT/examples/biosciences/evo2/data

OUTDIR="genome_data"
mkdir -p $OUTDIR
cd $OUTDIR

echo "download_path:" $(pwd)
wget -c https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr20.fa.gz
wget -c https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr21.fa.gz
wget -c https://hgdownload.soe.ucsc.edu/goldenpath/hg38/chromosomes/chr22.fa.gz

zcat chr20.fa.gz > chr20.fa
zcat chr21.fa.gz > chr21.fa
zcat chr22.fa.gz > chr22.fa

# 合并成一个文件
cat chr20.fa chr21.fa chr22.fa > chr20_21_22.fa

echo "数据准备完成：$OUTDIR/chr20_21_22.fa"

python $PROJECT_ROOT/examples/biosciences/evo2/tools/data_process/preprocess_data_fasta.py \
    --config $PROJECT_ROOT/examples/biosciences/evo2/config/genome_preprocess_config.yaml \