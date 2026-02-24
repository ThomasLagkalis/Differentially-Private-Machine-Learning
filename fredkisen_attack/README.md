# Model Iversion Attack (python)
Code implementing model inversion attack as described in [Fredkinsen et.al](https://dl.acm.org/doi/10.1145/2810103.2813677).

## Usage 

First, install all required packages with the following command:

```
pip install -r requirements.txt
```

The files which contain driver code for the experiments are the following:

* `avg_images.py`: Produces average images of each face in `./data/avg_images`.
* `main.py`: The main code for the experiments.
 
To see the flags of `main.py`  run it with `--help`, that is:

```
python main.py --help
```
