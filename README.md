
<h1 align="center">
  <a href="https://neurolifeblog.com">
    <img width="600" alt="Discovered ALife Simulations" src="https://pub.sakana.ai/asal_blog_assets/cover_video_square-min.png"></a><br>
</h1>


<h1 align="center">
Automating the Search for Artificial Life with Foundation Models
</h1>
<p align="center">
  <a href="https://neurolifeblog.com">Paper</a> |
  <a href="https://arxiv.org/abs/2412.17799">PDF</a>
</p>

[Lee Chase](https://x.com/leechase99)

## Abstract
With the recent Nobel Prize awarded for radical advances in protein discovery, foundation models (FMs) for exploring large combinatorial spaces promise to revolutionize many scientific fields. Artificial Life (ALife) has not yet integrated FMs, thus presenting a major opportunity for the field to alleviate the historical burden of relying chiefly on manual design and trial-and-error to discover the configurations of lifelike simulations. This paper presents, for the first time, a successful realization of this opportunity using vision-language FMs. The proposed approach, called *NeuroLife*, (1) finds simulations that produce target phenomena, (2) discovers simulations that generate temporally open-ended novelty, and (3) illuminates an entire space of interestingly diverse simulations. Because of the generality of FMs, NeuroLife works effectively across a diverse range of ALife substrates including Boids, Particle Life, Game of Life, Lenia, and Neural Cellular Automata. A major result highlighting the potential of this technique is the discovery of previously unseen Lenia and Boids lifeforms, as well as cellular automata that are open-ended like Conway's Game of Life. Additionally, the use of FMs allows for the quantification of previously qualitative phenomena in a human-aligned way. This new paradigm promises to accelerate ALife research beyond what is possible through human ingenuity alone.

<div style="display: flex; justify-content: space-between;">
  <img src="https://pub.sakana.ai/asal_blog_assets/teaser.png" alt="Image 1" style="width: 48%;">
  <img src="https://pub.sakana.ai/asal_blog_assets/methods_figure.png" alt="Image 2" style="width: 48%;">
</div>

## Repo Description
This repo contains a minimalistic implementation of NeuroLife to get you started ASAP.
Everything is implemented in the [Jax framework](https://github.com/jax-ml/jax), making everything end-to-end jittable and very fast.


The important code is here:
- [foundation_models/__init__.py](foundation_models/__init__.py) has the code to create a foundation model.
- [substrates/__init__.py](substrates/__init__.py) has the code to create a substrate.
- [rollout.py](rollout.py) has the code to rollout a simulation efficiently.
- [asal_metrics.py](asal_metrics.py) has the code to compute the metrics from NeuroLife.

Here is some minimal code to sample some random simulation parameters and run the simulation and evaluate how open-ended it is:
```python
import jax
from functools import partial
import substrates
import foundation_models
from rollout import rollout_simulation
import asal_metrics

fm = foundation_models.create_foundation_model('clip')
substrate = substrates.create_substrate('lenia')
rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=fm, rollout_steps=substrate.rollout_steps, time_sampling=8, img_size=224, return_state=False) # create the rollout function
rollout_fn = jax.jit(rollout_fn) # jit for speed
# now you can use rollout_fn as you need...
rng = jax.random.PRNGKey(0)
params = substrate.default_params(rng) # sample random parameters
rollout_data = rollout_fn(rng, params)
rgb = rollout_data['rgb'] # shape: (8, 224, 224, 3)
z = rollout_data['z'] # shape: (8, 512)
oe_score = asal_metrics.calc_open_endedness_score(z) # shape: ()
```

## ALife Substrates
We have already implemented the following ALife substrates:
- 'lenia': [Lenia](https://en.wikipedia.org/wiki/Lenia)
- 'boids': [Boids](https://en.wikipedia.org/wiki/Boids)
- 'plife': [Particle Life](https://www.youtube.com/watch?v=scvuli-zcRc)
- 'plife_plus': Particle Life++
  - (Particle Life with changing color dynamics)
- 'plenia': [Particle Lenia](https://google-research.github.io/self-organising-systems/particle-lenia/)
- 'dnca': Discrete Neural Cellular Automata
- 'nca_d1': [Continuous Neural Cellular Automata](https://distill.pub/2020/growing-ca/)
- 'gol': [Game of Life/Life-Like Cellular Automata](https://en.wikipedia.org/wiki/Life-like_cellular_automaton)

You can find the code for these substrates at [substrates/](substrates/)

## Running NeuroLife

The main files to run the entire NeuroLife pipeline are the following:

### Supervised Target & Open-Endedness
Use [main_opt.py](main_opt.py) with Sep-CMA-ES (from evosax):
```sh
# Search for a Lenia simulation matching a single prompt
python main_opt.py --seed=0 --save_dir="./data/supervised_0" --substrate='lenia' \
  --time_sampling=1 --prompts="a caterpillar" --coef_prompt=1. --coef_softmax=0. \
  --coef_oe=0. --bs=1 --pop_size=16 --n_iters=1000 --sigma=0.1

# Search for temporal targets (sequence of events)
python main_opt.py --seed=4 --save_dir="./data/supervised_temporal_4" --substrate='lenia' \
  --time_sampling=2 --prompts="a small biological cell;a large biological cell" \
  --coef_prompt=1. --coef_softmax=0.1 --coef_oe=0. --bs=1 --pop_size=16 --n_iters=1000 --sigma=0.1

# Search for open-ended simulations
python main_opt.py --seed=3 --save_dir="./data/open_endedness_3" --substrate='lenia' \
  --time_sampling=32 --prompts="" --coef_prompt=0. --coef_softmax=0. --coef_oe=1. \
  --bs=1 --pop_size=16 --n_iters=1000 --sigma=0.1

# Combine supervised target + open-endedness
python main_opt.py --seed=0 --save_dir="./data/st_oe_0" --substrate='lenia' \
  --time_sampling=32 --prompts="a diverse ecosystem of cells" --coef_prompt=1. \
  --coef_softmax=0. --coef_oe=1. --bs=1 --pop_size=16 --n_iters=1000 --sigma=0.1
```

### Illumination
Use [main_illuminate.py](main_illuminate.py) with a custom genetic algorithm:
```sh
python main_illuminate.py --seed=0 --save_dir="./data/illuminate_0" --substrate='lenia' \
  --n_child=32 --pop_size=256 --n_iters=1000 --sigma=0.1
```

### Game of Life Sweep
Use [main_sweep_gol.py](main_sweep_gol.py) with brute force search (discrete search space):
```sh
python main_sweep_gol.py
```

[neurolife.ipynb](neurolife.ipynb) goes through everything you need to know in detail, including how to visualize results.

## Tips for Getting Good Results

Some tips on getting the results you want from a given substrate:
- **Run for many seeds.** Meta optimizing chaotic systems is quite tough and noisy. More seeds can significantly increase the chances of getting what you want.
- **Play around with different hyperparameters.**
    - When doing the supervised temporal targets with multiple prompts, `coef_softmax` needs to be tuned.
    - Pay attention to the `time_sampling` variable and tune it. For open-endedness, try setting `time_sampling` to a large value (32 works well).
- **Run for more search iterations.** This can get you quite far sometimes.
- **Pick good expressive substrates.** Some substrates aren't capable of expressing certain lifeforms. Check out the discussion section of the paper.
- **Use better search algorithms!** We used Sep-CMA-ES, SGD, GAs, and brute force search in the paper, but other search algorithms may be better for certain substrates you choose.
- **CLIP seems to be a good choice of foundation model.** I don't think this is the bottleneck. But newer foundation models may be better than CLIP.

Particle Life++ has lots of potential and is an extremely untapped substrate so far, so experimenting with that would be very cool, although optimization is hard in that, due to how chaotic it is.

If you get bored, it would be amazing to apply NeuroLife to newer substrates like [ALIEN](https://www.youtube.com/watch?v=qwbMGPkoJmg) and [JaxLife](https://github.com/luchris429/jaxlife)!

## Running Locally
### Installation

To run this project locally, you can start by cloning this repo.
```sh
git clone https://github.com/neurolife-org/NeuroLife.git
```
Then, set up the python environment with conda:
```sh
conda create --name neurolife python=3.10.13
conda activate neurolife
```

Then, install the necessary python libraries:
```sh
python -m pip install -r requirements.txt
```
However, if you want GPU acceleration (trust me, you do), please [manually install jax](https://github.com/jax-ml/jax?tab=readme-ov-file#installation) according to your system's CUDA version.

### Loading Our Dataset of Simulations

We provide pre-computed datasets from our large scale searches:

- The Lenia and Boids dataset contains **8192 simulations** found in a large illumination search.
- The GoL dataset contains the **262,144 simulations** ranked in order of most open-ended to least open-ended.

You can view our dataset of simulations at:
- [Lenia Dataset](https://pub.sakana.ai/asal/data/illumination_poster_lenia.png)
- [Boids Dataset](https://pub.sakana.ai/asal/data/illumination_poster_boids.png)

You can download the datasets from:
```sh
wget -P ./data https://pub.sakana.ai/asal/data/illumination_lenia.npz
wget -P ./data https://pub.sakana.ai/asal/data/illumination_boids.npz
wget -P ./data https://pub.sakana.ai/asal/data/illumination_plife.npz
wget -P ./data https://pub.sakana.ai/asal/data/sweep_gol.npz
```

Here's how to load and visualize a specific simulation from the dataset:
```python
import numpy as np
from functools import partial
import jax
import substrates
from rollout import rollout_simulation

params_all = np.load("./data/illumination_lenia.npz", allow_pickle=True)['params']
print(params_all.shape) # shape: (number of simulations, parameter dimension)
params = params_all[6198] # visualize simulation #6198

substrate = substrates.create_substrate('lenia')
substrate = substrates.FlattenSubstrateParameters(substrate)

rollout_fn = partial(rollout_simulation, s0=None, substrate=substrate, fm=None,
                     rollout_steps=substrate.rollout_steps, time_sampling=8,
                     img_size=224, return_state=False)
rollout_fn = jax.jit(rollout_fn)

rng = jax.random.PRNGKey(0)
rollout_data = rollout_fn(rng, params)
# rollout_data['rgb'] has shape (8, 224, 224, 3)
```

Directions on how to visualize and load these simulations in more detail are shown in [neurolife.ipynb](neurolife.ipynb).

## Reproducing Results from the Paper
Everything you need is already in this repo.

## Bibtex Citation
To cite our work, you can use the following:
```
@article{chase2024neurolife,
  title = {Automating the Search for Artificial Life with Foundation Models},
  author = {Lee Chase},
  year = {2024},
  url = {https://neurolifeblog.com}
}
```
