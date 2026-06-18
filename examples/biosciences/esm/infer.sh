  # 1. ESM2 提取序列表征

  python examples/biosciences/esm/scripts/extract.py \
    $ONESCIENCE_MODELS_DIR/esm_models/esm2_t6_8M_UR50D.pt \
    examples/biosciences/esm/data/few_proteins.fasta \
    /tmp/esm_extract_out \
    --include mean per_tok \
    --repr_layers 6

  # 2. ESMFold 结构推理

  python examples/biosciences/esm/scripts/fold.py \
    -i examples/biosciences/esm/data/few_proteins.fasta \
    -o /tmp/esmfold_pdb_out \
    --model-dir $ONESCIENCE_MODELS_DIR/esm_models/

  # 3. 变异打分推理
  python examples/biosciences/esm/variant-prediction/predict.py \
    --model-location esm1v_t33_650M_UR90S_1 \
    --sequence WTSEQUENCE_HERE \
    --dms-input examples/biosciences/esm/variant-prediction/data/BLAT_ECOLX_Ranganathan2015.csv \
    --mutation-col mutant \
    --dms-output /tmp/esm_variant_prediction.csv \
    --offset-idx 24 \
    --scoring-strategy wt-marginals