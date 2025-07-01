# BayesDiff: A Bayesian Nonparametric Approach to Differential Analysis of Genomic Data

# TODO:

> NOTE: if we want to use UVIT which I prefer we still have to deal with the mess coming from latent space
> To deal with it you do Monte Carlo sampling, which allows you to estimate epistemic uncertainty

- This is added as the [openai_unet](https://github.com/openai/guided-diffusion/tree/main) it has been simplified and classifier free guidance applied

### Added but to review

- [ ] Fix diffiusion DDIM
- [ ] Improve repository structure
- [ ] Flow Matching Review where Insereting FILM conditioning inside all the model for more stability [chat](https://chatgpt.com/share/686007aa-c180-8008-ae25-ba41169ff163)

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
python -m src.train.run --epochs 4 --method=diffusion
```

This will log results to wanbd if enabled. Keep in mind that you have to add your api key inside a `.env` file in the root of the repo. You can take inspiration from `.env_sample`

- You can also run a model to generate images:
  > The following is an example which use the best_model checkpoint to generate images inside the checkpoint directory by default

```bash
python -m src.eval.generate --n 10
```

### Additional results

Write some code inspired by the paper to use a better pre-trained model to showcase some nice results

# NOTE:

> [!WARNING]
> THE rest of the README is just GPT stuff for the future

## 📘 Project Overview

This is the final project for the **Probabilistic Machine Learning and Deep Learning** course. In this project, we explore, analyze, and re-implement the methodology presented in the paper:

> **BayesDiff: A Bayesian Nonparametric Approach to Differential Analysis of Genomic Data**
>
> [Zhu, J., Ibrahim, J.G., & Love, M.I. (2019). Bioinformatics.](https://academic.oup.com/bioinformatics/article/35/21/4447/5480446)

The BayesDiff framework leverages Bayesian nonparametric techniques to perform differential analysis of high-throughput genomic data, focusing on identifying differentially expressed regions with spatial dependencies.

## 🛠️ Project Components

...
