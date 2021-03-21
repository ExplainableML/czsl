# Compositional Graph Embedding
Code to reproduce experiments for Compositional Graph Embedding

1. Clone the repo

2. We recommend using Anaconda for environment setup. The additional required packages are:
```
    conda install pytorch torchvision torchaudio cudatoolkit=11.0 -c pytorch
    pip install tqdm
    pip install tensorboard
```

3. Go to the cloned repo and open a terminal. Download the dataset.
```
    bash ./utils/download_data.sh
    mkdir logs
```

4. Reformat MIT-States folder structure.
```
    cd data/mit-states/images
    sudo apt install rename
    rename 's/ /_/g' *
```
    
4. Run training for MIT-States.
```
    python train.py --config configs/mit.yml
```

5. Run training for UT-Zappos.
```
    python train.py --config configs/utzappos.yml
```

6. Run a jupyter notebook for visualizing results on the test set. The notebook is located in notebooks/analysis.ipynb
```
    jupyter notebook
```

Note: Some of the scripts are adapted from AttributeasOperators repository. GCN and GCNII implementations are imported from their respective repositories.

## License
taskmodularnets is Creative Commons Attribution Non-Commercial Licensed, as found in the LICENSE file.
