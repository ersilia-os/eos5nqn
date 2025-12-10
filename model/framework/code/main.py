# imports
import os
import csv
import sys
import subprocess
import tempfile

# variables
root = os.path.dirname(os.path.abspath(__file__))
checkpoints_dir = os.path.abspath(os.path.join(root, "..", "..", "checkpoints"))
python_executable = sys.executable

# parse arguments
input_file = sys.argv[1]
output_file = sys.argv[2]

# current file directory
root = os.path.dirname(os.path.abspath(__file__))
cwd = os.getcwd()

# read SMILES from .csv file, assuming one column with header
with open(input_file, "r") as f:
    reader = csv.reader(f)
    next(reader)
    smiles_list = [r[0] for r in reader]

# make temporary folder
tmp_dir = os.path.abspath(tempfile.mkdtemp(prefix="ersilia-"))
tmp_input = os.path.join(tmp_dir, "input.csv")
with open(tmp_input, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["SMILES"])
    for smiles in smiles_list:
        writer.writerow([smiles])
tmp_bash = os.path.join(tmp_dir, "runner.sh")
tmp_output = os.path.join(tmp_dir, "predictions.csv")
content = f"""
cd {root} 

{python_executable} gneprop_predict.py --dataset_path {tmp_input} \
                    --supervised_pretrain_path {checkpoints_dir}/20250819-093608/ \
                    --target_name DummyTarget \
                    --output_path {tmp_output}

cd {cwd}
"""
with open(tmp_bash, "w") as f:
    f.write(content)

# run model
process = subprocess.Popen(f"bash {tmp_bash}", shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
stdout, stderr = process.communicate()
if stdout:
    print("Subprocess output:")
    print(stdout)
if stderr:
    print("Subprocess errors:")
    print(stderr)

# collect output
outputs = []
with open(tmp_output, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for r in reader:
        outputs += [float(r[1])]

# check input and output have the same lenght
input_len = len(smiles_list)
output_len = len(outputs)
assert input_len == output_len

# write output in a .csv file
with open(output_file, "w") as f:
    writer = csv.writer(f)
    writer.writerow(["tolc_activity"])  # header
    for o in outputs:
        writer.writerow([o])
