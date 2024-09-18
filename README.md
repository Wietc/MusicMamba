# MusicMamba

This is the official implementation of MusicMamba.

*Checkout our demo and paper* : [Demo](https://moersxm.github.io/MusicMamba_Demo/) | [arXiv](https://arxiv.org/abs/2409.02421)

## Environment
* Clone this Repo 

    ```bash
    git clone https://github.com/Wietc/MusicMamba.git
    ```

* using python version 3.11.5
* using pytorch version 2.2.1
* install python dependencies

    `pip install -r requirements.txt`

* Mamba needs to be downloaded separately
  
    `pip install mamba_ssm`

* install checkpoints from Huggingface
  
    `https://huggingface.co/moersxm12138/MusicMamba`

## To train the model with GPU

We currently do not offer fine-tuning functionality.

## To generate music

`python generate.py`

* The specification of the model path, data path, and other generated parameters are in the `utilities/argument_funcs.py` file.

##  Details of the files in this repo
```
`
├── data                    Stores train, test and val data.
│   └── FolkDB              
│       ├── train
│       ├── test
│       └── val
├── dataset.py              Progress datasets.
├── generate.py             For generating music. (Detailed usage are written in the file)
├── model.py                The MusicMamba Architecture.
├── midi_tokenize           Remi-M tokenize.
├── utilities               Tools for generating music.
│   ├── argument_funcs.py   Some arguments for generating.
│   ├── constants.py        
│   └── device.py           
└── README.md               Readme
```

## Citation
If you find this work helpful and use our code in your research, please kindly cite our paper:
```
@article{MusicMamba,
title={MusicMamba: A Dual-Feature Modeling Approach for Generating Chinese Traditional Music with Modal Precision},
author={Jiatao Chen and Xing Tang and Tianming Xie and Jing Wang and Wenjing Dong and Bing Shi}, year={2024},
eprint={2409.02421},
archivePrefix={arXiv},
}
```
