# run_mass_generation.sh
#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=.

# Get test set size
N=$(python - <<'PY'
from mdmad.utils.misc import load_config
from mdmad.datasets import get_dataset
cfg,_ = load_config('./configs/test/codesign_single.yml')
print(len(get_dataset(cfg.dataset.test)))
PY
)
echo "Test set size: $N"
for idx in $(seq 0 $((N-1))); do
  python design_testset.py $idx \
    --config ./configs/test/mdmad_k1_k1.yml \
    --out_root ./results_mdmad_k1_k1 \
    --device cuda --batch_size 16 --seed 2022
done

python -m mdmad.tools.relax \
  --root "./results_mdmad_k1_k1" \

python -m mdmad.tools.eval \
  --root "./results_mdmad_k1_k1" \