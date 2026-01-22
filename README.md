# MDMAD

![cover-large](./assets/mdmad.pdf)

Mixture Diffusion Model for Multimodal Antibody Design 

## Install

### Environment

```bash
conda env create -f env.yaml -n mdmad
conda activate mdmad
```

set ur toolkit version in [`env.yaml`](./env.yaml).

### Datasets and Trained Weights

Protein structures in the `SAbDab` dataset can be downloaded [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/). Extract `all_structures.zip` into the `data` folder. 

The `data` folder contains a snapshot of the dataset index (`sabdab_summary_all.tsv`). You may replace the index with the latest version [**here**](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/summary/all/).

Trained model weights are available [**here** (Google Drive)](https://drive.google.com/drive/folders/15ANqouWRTG2UmQS_p0ErSsrKsU4HmNQc?usp=sharing).

### PyRosetta

PyRosetta is required to relax the generated structures and compute binding energy. Please follow the instruction [**here**](https://www.pyrosetta.org/downloads) to install.

### Ray

Ray is required to relax and evaluate the generated antibodies. Please install Ray using the following command:

```bash
pip install -U ray
```

## Train mdmad (example for Kx = 1, Ko = 1

```bash
python train.py ./configs/train/mdmad_k1_k1.yml
```

| Config File              | Description                                                  |
| ------------------------ | ------------------------------------------------------------ |
| `mdmad.yml`              | Sample both the **sequence** and **structure** of **one** CDR. |


## Sampling (example for Kx = 1, Ko = 1)
First configure checkpoint file path in ./configs/test/mdmad_k1_k1.yml, as well as the bash file (run_mass_generation_mdmad_k1_k1.sh)
```bash
chmod +x run_mass_generation_mdmad_k1_k1.sh
./run_mass_generation_mdmad_k1_k1.sh
```

## Evaluation
python -m mdmad.tools.eval 

