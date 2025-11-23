# Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs

Here is the official PyTorch implementation for the paper **"Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs"**, which has been accepted by **KDD '25**.

For more details, please refer to https://dl.acm.org/doi/10.1145/3711896.3737156

This project proposes a novel framework **TGODE** to address two critical factors often overlooked in existing methods: **irregular user interests** and **highly uneven item distributions**. TGODE constructs two tailored graphs—a user time graph and an item evolution graph—and utilizes a time-guided diffusion generator alongside a generalized graph neural ODE to align user preferences and item trends over continuous time.

**Authors**: Haoyan Fu, Zhida Qin, Shixiao Yang, Haoyao Zhang, Bin Lu, Shuang Li, Tianyu Huang, and John C.S. Lui.
**Affiliation**: Beijing Institute of Technology, Shanghai Jiao Tong University, Beihang University, The Chinese University of Hong Kong.

## Architecture

The overall architecture of TGODE consists of three main parts: **Pivotal Graph Generation**, **Time-Guided Diffusion Generator**, and **Generalized Graph Neural ODEs**.

![https://github.com/Qin-lab-code/TGODE/blob/main/main.png]()


## Requirements

The code is implemented using **PyTorch**. The mainly required packages are listed below:

```bash
python>=3.8
torch>=1.10.0
numpy>=1.20.3
scipy>=1.6.2
torchdiffeq>=0.2.0  # For ODE solvers
```

## Usage

<ol> <li>Data Preparation: Download the datasets from <a href="https://jmcauley.ucsd.edu/data/amazon/">Amazon Review Data</a> and MovieLens, and place them in the <code>dataset/</code> directory.</li>  <li>Training: Run the main script to train and evaluate the model:</li> </ol>

```bash
python main.py --model TGODE --dataset Beauty
```

## Implemented Models

<table class="table table-hover table-bordered"> <tr> <th>Model</th>         <th>Paper</th>      <th>Type</th>   <th>Code</th> </tr> <tr> <td scope="row">TGODE</td> <td>Fu et al. <a href="https://doi.org/10.1145/3711896.3737156" target="_blank">Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs</a>, KDD '25. </td> <td>GNN + ODE + Diffusion</td> <td><a href="https://github.com/fhy99/TGODE">PyTorch</a> </td> </tr> </table>

## Related Datasets

We conduct extensive experiments on five real-world datasets. The statistics are summarized below :

| **Datasets**  | **Beauty** | **Sports** | **Toys**  | **Video** | **ML-100k** |
| ------------- | ---------- | ---------- | --------- | --------- | ----------- |
| # User        | $22,363$   | $35,598$   | $19,412$  | $24,303$  | $943$       |
| # Item        | $12,101$   | $18,357$   | $11,924$  | $10,672$  | $1,682$     |
| # Interaction | $198,502$  | $296,337$  | $167,597$ | $231,780$ | $100,000$   |
| # Avg. Len    | $8.87$     | $8.32$     | $8.63$    | $9.53$    | $106.04$    |
| Density       | $99.92\%$  | $99.95\%$  | $99.92\%$ | $99.91\%$ | $93.69\%$   |

## Reference

If you find this repo helpful to your research, please cite our paper :

```BibTeX
@inproceedings{fu2025time,
  title={Time Matters: Enhancing Sequential Recommendations with Time-Guided Graph Neural ODEs},
  author={Fu, Haoyan and Qin, Zhida and Yang, Shixiao and Zhang, Haoyao and Lu, Bin and Li, Shuang and Huang, Tianyu and Lui, John CS},
  booktitle={Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2},
  pages={637--648},
  year={2025}
}

```


