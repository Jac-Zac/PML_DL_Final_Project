# BayesDiff: A Bayesian Nonparametric Approach to Differential Analysis of Genomic Data

# TODO:

> NOTE: if we want to use UVIT which I prefer we still have to deal with the mess coming from latent space
> To deal with it you do Monte Carlo sampling, which allows you to estimate epistemic uncertaint

- Modify the model to have a linear layer at the end
- Modify the validation to use the fast sampling instead (if you want). We can avoid fast sampling
- Try to do LLLA after you have changed the last layer to be linear
- Implement Flow Matching instead of Diffiusion (using still the same Unet), We should be able to swap with a model registry

### Added but to review

- [ ] Flow Matching
- [ ] LLM model

## How to run the project:

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

### 1. **Paper Review**

We provide a thorough review of the BayesDiff paper, explaining key ideas, models, and inference techniques, including:

- Hierarchical Bayesian modeling
- Dirichlet Process Mixtures
- MCMC sampling strategies
- Handling spatial dependencies in genomic data

### 2. **Reimplementation**

We re-implemented the core BayesDiff model using Python and PyMC. Our implementation includes:

- Data preprocessing pipeline for gene expression data
- Model construction with Bayesian nonparametric priors
- Gibbs and Metropolis-Hastings samplers
- Posterior inference and diagnostics

### 3. **Experiments**

We ran extensive experiments on synthetic and real datasets:

- Evaluated performance in detecting differentially expressed regions
- Compared BayesDiff to baseline methods such as DESeq2 and edgeR
- Visualized posterior distributions and region detection accuracy

## 📁 Repository Structure

```

BayesDiff-Project/
├── data/ # Sample synthetic and real datasets
├── notebooks/ # Jupyter notebooks for analysis and visualization
├── src/ # Core implementation of the BayesDiff model
│ ├── model.py # Model definition
│ ├── inference.py # Inference algorithms
│ └── utils.py # Helper functions
├── results/ # Experimental results and plots
├── report/ # Final written report (PDF)
└── README.md # Project documentation

```

## 🔍 Key Results

- BayesDiff outperforms standard methods in regions with high spatial correlation.
- Posterior uncertainty quantification adds interpretability to differential expression analysis.
- The nonparametric nature allows flexibility in modeling unknown group structures.

## 📈 Visualizations

We include various plots:

- Trace plots and posterior diagnostics
- Detected differential regions across the genome
- Comparisons with true differential regions (synthetic data)

## 📚 Dependencies

- Python 3.9+
- NumPy, SciPy, pandas
- PyMC 4+
- ArviZ
- Matplotlib, Seaborn

Install dependencies:

```bash
pip install -r requirements.txt
```

## 🧪 Running the Project

> Run the following commands inside the root of the directory

To train the model:

```bash
python -m src.train.run --epochs 3
```

To sample an image and view the different timesteps :

> Saving the otutput in a custom directory

```bash
python -m src.eval.generate --n 4 --ckpt checkpoints/best_model.pth
```

## 👨‍👩‍👧‍👦 Contributors

- Your Name
- Collaborator Name(s)
  (Course: Probabilistic Machine Learning and Deep Learning, 2025)

## 📄 License

This project is for academic and educational purposes.

---

Feel free to clone, explore, and adapt the BayesDiff model for your own datasets!

```

```
