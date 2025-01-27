# DNA Playground

## Usage

```bash
conda create -n dna python=3.9
conda activate dna
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

git clone https://github.com/luxixi2021/EBMGP.git
cd dna-playground
pip install -v -e .

# train
python tools/train.py fit --config configs/htrans.yaml
```
