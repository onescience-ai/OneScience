import sys
import numpy as np
from math import pow
# import py3Dmol
from pathlib import Path
from io import StringIO
from Bio.PDB import PDBIO
from Bio.PDB import MMCIFParser, Superimposer
import time

from onescience.models.simplefold.wrapper import ModelWrapper, InferenceWrapper

# sys.path.append(str(Path("./src/simplefold").resolve()))


# following are example amino acid sequences:
example_sequences = {
    "7ftv_A": "GASKLRAVLEKLKLSRDDISTAAGMVKGVVDHLLLRLKCDSAFRGVGLLNTGSYYEHVKISAPNEFDVMFKLEVPRIQLEEYSNTRAYYFVKFKRNPKENPLSQFLEGEILSASKMLSKFRKIIKEEINDDTDVIMKRKRGGSPAVTLLISEKISVDITLALESKSSWPASTQEGLRIQNWLSAKVRKQLRLKPFYLVPKHAEETWRLSFSHIEKEILNNHGKSKTCCENKEEKCCRKDCLKLMKYLLEQLKERFKDKKHLDKFSSYHVKTAFFHVCTQNPQDSQWDRKDLGLCFDNCVTYFLQCLRTEKLENYFIPEFNLFSSNLIDKRSKEFLTKQIEYERNNEFPVFD",
    "8cny_A": "MGPSLDFALSLLRRNIRQVQTDQGHFTMLGVRDRLAVLPRHSQPGKTIWVEHKLINILDAVELVDEQGVNLELTLVTLDTNEKFRDITKFIPENISAASDATLVINTEHMPSMFVPVGDVVQYGFLNLSGKPTHRTMMYNFPTKAGQCGGVVTSVGKVIGIHIGGNGRQGFCAGLKRSYFAS",
    "8g8r_A": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
    "8i85_A": "MGILQANRVLLSRLLPGVEPEGLTVRHGQFHQVVIASDRVVCLPRTAAAAARLPRRAAVMRVLAGLDLGCRTPRPLCEGSLPFLVLSRVPGAPLEADALEDSKVAEVVAAQYVTLLSGLASAGADEKVRAALPAPQGRWRQFAADVRAELFPLMSDGGCRQAERELAALDSLPDITEAVVHGNLGAENVLWVRDDGLPRLSGVIDWDEVSIGDPAEDLAAIGAGYGKDFLDQVLTLGGWSDRRMATRIATIRATFALQQALSACRDGDEEELADGLTGYR",
    # "8g8r_A_x": "GTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFSGTVNWSVEDIVKGINSNNLESQLQATQAARKLLSREKQPPIDNIIRAGLIPKFVSFLGKTDCSPIQFESAWALTNIASGTSEQTKAVVDGGAIPAFISLLASPHAHISEQAVWALGNIAGDGSAFRDLVIKHGAIDPLLALLAVPDLSTLACGYLRNLTWTLSNLCRNKNPAPPLDAVEQILPTLVRLLHHNDPEVLADSCWAISYLTDGPNERIEMVVKKGVVPQLVKLLGATELPIVTPALRAIGNIVTGTDEQTQKVIDAGALAVFPSLLTNPKTNIQKEATWTMSNITAGRQDQIQQVVNHGLVPFLVGVLSKADFKTQKEAAWAITNYTSGGTVEQIVYLVHCGIIEPLMNLLSAKDTKIIQVILDAISNIFQAAEKLGETEKLSIMIEECGGLDKIEALQRHENESVYKASLNLIEKYFS",
}




simplefold_model = "simplefold_3B" # choose from 100M, 360M, 700M, 1.1B, 1.6B, 3B
backend = "torch" # choose from ["mlx", "torch"]
ckpt_dir = "artifacts"
output_dir = "artifacts"
prediction_dir = f"predictions_{simplefold_model}_{backend}"
# output_name = f"{seq_id}"
num_steps = 500 # number of inference steps for flow-matching
tau = 0.05 # stochasticity scale
plddt = True # whether to use pLDDT confidence module
nsample_per_protein = 1 # number of samples per protein

# ESM模型离线模式配置
# offline_mode = False  # 设置为True以使用离线模式
# esm_model_dir = "/public/home/onescience2025404/zhangyq/ml-simplefold-main/esm_models/esm2_t36_3B_UR50D"  # 离线模式下ESM模型的本地路径

# 如果在计算节点上没有网络，请设置:
# offline_mode = True
# esm_model_dir = "/path/to/your/downloaded/esm/models"
# 
# 注意: 
# 1. esm_model_dir 应该指向包含模型文件的目录
# 2. 支持两种格式:
#    - ESM原生格式: 包含 {model_name}.pt 文件
#    - Hugging Face格式: 包含 config.json 和 pytorch_model.bin.index.json 文件
# 3. 当前配置使用的是Hugging Face格式的模型


# initialize the folding model and pLDDT model
model_wrapper = ModelWrapper(
    simplefold_model=simplefold_model,
    ckpt_dir=ckpt_dir,
    plddt=plddt,
    backend=backend,
)
device = model_wrapper.device
folding_model = model_wrapper.from_pretrained_folding_model()
plddt_model = model_wrapper.from_pretrained_plddt_model()



# initialize the inference module with inference configurations
inference_wrapper = InferenceWrapper(
    output_dir=output_dir,
    prediction_dir=prediction_dir,
    num_steps=num_steps,
    tau=tau,
    nsample_per_protein=nsample_per_protein,
    device=device,
    backend=backend,
)


"""
Batch inference over all example sequences.

This will iterate through example_sequences, run inference for each, and
save the predicted structures under the configured prediction directory.
The visualization and TM-score sections below are skipped after this loop.
"""

all_save_paths = {}
for seq_id, aa_sequence in example_sequences.items():
    print(f"\n==== Predicting structure for {seq_id} with {len(aa_sequence)} amino acids ====")
    output_name = f"{seq_id}"

    batch, structure, record = inference_wrapper.process_input(aa_sequence)
    
    inference_start_time = time.time()
    results = inference_wrapper.run_inference(
        batch,
        folding_model,
        plddt_model,
        device=device,
    )
    print(
        f'  Running model inference took'
        f' {time.time() - inference_start_time:.2f} seconds.'
    )
    save_paths = inference_wrapper.save_result(
        structure,
        record,
        results,
        out_name=output_name,
    )
    all_save_paths[seq_id] = save_paths
    print(f"Saved {len(save_paths)} structure(s) for {seq_id}:")
    for p in save_paths:
        print(f" - {p}")
        try:
            # compute mean pLDDT from B-factors of CA atoms
            parser_tmp = MMCIFParser(QUIET=True)
            st = parser_tmp.get_structure("pred", p)
            b_factors = [a.get_bfactor() for a in st.get_atoms() if a.get_id() == 'CA']
            if len(b_factors) > 0:
                mean_plddt = float(np.mean(b_factors))
                print(f"   mean pLDDT: {mean_plddt:.2f}")
        except Exception as e:
            print(f"   pLDDT parse failed: {e}")

        # try to compute TM-score and RMSD if ground truth exists
        try:
            ref_path = Path(f"assets/{seq_id}.cif")
            if ref_path.exists():
                parser_ref = MMCIFParser(QUIET=True)
                struct_ref = parser_ref.get_structure("ref", str(ref_path))
                struct_prd = parser_ref.get_structure("prd", p)

                atoms_ref = [a for a in struct_ref.get_atoms() if a.get_id() == 'CA']
                atoms_prd = [a for a in struct_prd.get_atoms() if a.get_id() == 'CA']
                if len(atoms_ref) == 0 or len(atoms_prd) == 0:
                    print("   TM-score skipped: no CA atoms found")
                else:
                    # superimpose predicted onto reference using CA atoms
                    sup = Superimposer()
                    # use min length to avoid mismatch
                    n = min(len(atoms_ref), len(atoms_prd))
                    sup.set_atoms(atoms_ref[:n], atoms_prd[:n])
                    sup.apply(struct_prd.get_atoms())

                    coords_ref = np.array([a.coord for a in atoms_ref[:n]])
                    coords_prd = np.array([a.coord for a in atoms_prd[:n]])

                    # TM-score calculation (same formula as below section)
                    L_target = coords_ref.shape[0]
                    dists = np.linalg.norm(coords_ref - coords_prd, axis=1)
                    d0 = 1.24 * pow(L_target - 15, 1/3) - 1.8
                    if d0 < 0.5:
                        d0 = 0.5
                    tm_score = float(np.sum(1.0 / (1.0 + (dists / d0) ** 2)) / L_target)

                    print(f"   TM-score: {tm_score:.3f}")
                    print(f"   RMSD: {sup.rms:.3f}")
            else:
                print("   TM-score skipped: ground truth CIF not found under assets/")
        except Exception as e:
            print(f"   TM-score computation failed: {e}")

print("\nAll predictions completed.")
print("Summary of outputs:")
for seq_id, paths in all_save_paths.items():
    print(f" {seq_id}:")
    for p in paths:
        print(f"   - {p}")
