# BayesFlow: Extension of BayesDiff to Flow matching

... Some nice plots here ...

[Training Runs Report](https://api.wandb.ai/links/jac-zac/h0ack55v)

## 🧪 Running the Project

- Create the virtual environment

```bash
 python3.12 -m venv .venv
```

- Activate the environment

```bash
source .venv/bin/activate
```

- Install dependencies

```bash
pip install -r requirements.txt
```

- Train the model
  > From the root of the directory

The following is just an example to get all options you can run: `python -m src.train.run --help`

```bash
python -m src.train.run --epochs 20 --method=diffusion
```

This will log results to wanbd if enabled. Keep in mind that you have to add your api key inside a `.env` file in the root of the repo. You can take inspiration from `.env_sample`

- You can also run a model to generate images:
  > The following is an example which use the best_model checkpoint to generate images inside the checkpoint directory by default

```bash
python -m src.eval.generate --n 10
```

### To get all the plots

> You first need to train the model and then run the general script

```bash
./generate_llla_samples.sh
```

### Project Structure

```bash
.
├── notebooks # Notebooks for experiments
│   ├── notebook_llla_diff.ipynb
│   ├── notebook_llla_flow.ipynb
│   ├── notebook_train_diff.ipynb
│   └── notebook_train_flow.ipynb
├── src # Main part of the code
│   ├── eval
│   │   ├── generate.py
│   │   └── llla.py # Code to perform llla fit and make plots with it
│   ├── models
│   │   ├── __init__.py
│   │   ├── diffusion.py
│   │   ├── flow.py
│   │   ├── llla_model.py # Wrapper of the unet with LLLA
│   │   └── unet.py
│   ├── train # Train code for Flow/Diff on MNIST/Fashion-MNIST
│   │   ├── __init__.py
│   │   ├── run.py
│   │   └── train.py
│   ├── utils # Utilities for model training logging, loading and plots
│   │   ├── __init__.py
│   │   ├── data.py
│   │   ├── environment.py
│   │   ├── plots.py
│   │   └── wandb.py
│   └── __init__.py
├── generate_llla_samples.sh # Script to generate plots
├── notes.md # Additional Notes
├── README.md
└── requirements.txt
```

## References

- _Kou, S., Gan, L., Wang, D., Li, C., & Deng, Z. (2023). BayesDiff: Estimating Pixel-wise Uncertainty in Diffusion via Bayesian Inference. arXiv:2310.11142_

- _Daxberger, E., Kristiadi, A., Immer, A., Eschenhagen, R., Bauer, M., & Hennig, P. (2022). Laplace Redux -- Effortless Bayesian Deep Learning. arXiv:2106.14806_

- _Lipman, Y., Havasi, M., Holderrieth, P., Shaul, N., Le, M., Karrer, B., Chen, R. T. Q., Lopez-Paz, D., Ben-Hamu, H., & Gat, I. (2024). Flow Matching Guide and Code. arXiv:2412.06264_

- _Kristiadi, A., Hein, M., & Hennig, P. (2020). Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks [Conference presentation]. arXiv:2002.10118_
