# [Data Talks Club] GitOps for ML: Converting Notebooks to Reproducible Pipelines

In this hands-on workshop, we’ll take a prototype in a Jupyter Notebook and
transform it into a DVC pipeline. We’ll then use that pipeline locally to run
and compare a few experiments. Lastly, we’ll explore how CML will allow us to
take our model training online. We’ll use it in conjunction with GitHub Actions
to trigger our model training every time we push changes to our repository.

As an example project we'll use a Jupyter Notebook that trains a CNN to classify
images of Pokémon. It will predict whether a Pokémon is of a predetermined type
(default: water). It is a starting point that shows how a notebook might look
before it is transformed into a DVC pipeline.

It is a fork of this example project:
https://github.com/iterative/example-pokemon-classifier

_Note: due to the limited size of the dataset, the evaluation dataset is the
same data set as the train+test. Take the results of the model with a grain of
salt._

### Requirements

- [Python >= 3.9.13](https://www.python.org/downloads/)
- [Virtualenv >=
  20.14.1](https://virtualenv.pypa.io/en/latest/installation.html)

### Getting started

1. Fork the repository and clone it to your local environment

2. Create a new virtual environment with `virtualenv -p python3 .venv`

3. Activate the virtual environment with `source .venv/bin/activate`

4. Install the dependencies with `pip install -r requirements.txt`

5. Download the datasets from Kaggle into the `data/external/` directory

   - [data/external/pokemon-gen-1-8](https://www.kaggle.com/datasets/robdewit/pokemon-images)
   - [data/external/stats/pokemon-gen-1-8.csv](https://www.kaggle.com/datasets/rounakbanik/pokemon)

8. Launch the notebook with `jupyter-notebook` and open
   `pokemon_classifier.ipynb`

### Notes on hardware

The requirements specify `tensorflow-macos` and `tensorflow-metal`, which are
the appropriate requirements when you are using a Mac with an M1 CPU or later.
In case you are using a different system, you will need to replace these with
`tensorflow`.

## Workshop - part 1

Now that we have the notebook up and running, go through the cells to see if
everything works. If it does, you should get a model that generates predictions
for all Pokémon images. Although admittedly the model performance isn't great...

This point may be familiar to you: a working prototype in a notebook. Now, how
do we transform it into a reproducible DVC pipeline?

### Setting up DVC and tracking data

1. Initialize DVC with `dvc init`
2. Start tracking the `data` and `outputs` directories with DVC (`dvc add`)
3. Poke around with `git status` and see what DVC did in the background. Take a
   look at `data.dvc` or `outputs.dvc` to see the metadata files thta DVC
   created
4. Commit the changes to Git (`git commit -m "Start tracking data and outputs
   directories with DVC"`)

Now that the data is part of the DVC cache, we can set up a remote for
duplicating it. Just like we `git push` our local Git repository to GitHub,
Gitlab, etc., we can then `dvc push` our cache to the remote.

3. Use `dvc remote add` to add your remote of choice
   ([docs](https://dvc.org/doc/command-reference/remote/add))
   
4. Push the DVC cache to your remote with `dvc push`

### Create `params.yaml`

Once we start experimenting, we want to change parameters on the fly. For this,
we define a `params.yaml` file. Create this in the root directory of the
project. For example:

```yaml
base:
  seed: 42
  pokemon_type_train: "Water"

data_preprocess:
  source_directory: 'data/external'
  destination_directory: 'data/processed'
  dataset_labels: 'stats/pokemon-gen-1-8.csv'
  dataset_images: 'images-gen-1-8'

train:
  test_size: 0.2
  learning_rate: 0.001
  epochs: 15
  batch_size: 120
```

### Create Python modules

Now it is time to move out of our familiar notebook environment. We will split
up the notebook into units that make sense as a step in a pipeline. In this
case, we will create four stages: `data-preprocess`, `data_load`, `train`, and
`test`.

1. Create an `src` directory to move the modules to.
1. Create a `.py` file in the `src` directory for every pipeline step (e.g.
   `train.py`)
1. Copy the relevant code over to each module. Make sure to also include the
   imports needed in each section.
1. Create a `main` function so that we can call the module using a command.
   We'll use `argparse` so that we can pass our parameters:

  ```python
  if __name__ == '__main__':

      args_parser = argparse.ArgumentParser()
      args_parser.add_argument('--params', dest='params', required=True)
      args = args_parser.parse_args()

      with open(args.params) as param_file:
          params = yaml.safe_load(param_file)
  ```

1. Replace hardcoded parameters with references to `params.yaml`, for example:

  ```python
  MODEL_LEARNING_RATE: float = params['train']['learning_rate']
  MODEL_EPOCHS: int = params['train']['epochs']
  MODEL_BATCH_SIZE: int = params['train']['batch_size']
  ...
  estimator = model.fit(X_train, y_train, 
                        validation_data=(X_test, y_test),
                        class_weight= calculate_class_weights(y_train),
                        epochs=MODEL_EPOCHS, 
                        batch_size=MODEL_BATCH_SIZE,
                        verbose=1)
  ```

Once we're done, we should be able to run the module from your command line:
`python3 src/train.py --params params.yaml`. 

[If you'd like an example, check my implementation for `train.py`
here](https://github.com/iterative/example-pokemon-classifier/blob/main/src/train.py).

### Create pipeline
Just like we could run the cells in our notebook one-by-one, we can now run the
modules successively from our command line. But we can also create a `dvc.yaml`
file that defines a pipeline for us. We can then run the entire pipeline with a
single command.

1. Create a `dvc.yaml` file and set up the stages, their dependencies, and
   outputs
   ([docs](https://dvc.org/doc/user-guide/project-structure/dvcyaml-files#stages)).
1. Reproduce the pipeline with `dvc repro`.

[If you'd like an example, check my implementation for `dvc.yaml`
here](https://github.com/iterative/example-pokemon-classifier/blob/main/dvc.yaml)

### Run experiments

With our pipeline in place, we cannot only reproduce a pipeline run with a
single command; we can also run entirely new experiments. Let's explore two
ways:

1. Update a parameter in `params.yaml` (for example: `type: 'dragon'`) and use
   `dvc repro` to trigger a new model training
2. Run a new experiment with `dvc exp run` and use the `-S` option to set a
   parameter (for example: `dvc exp run -S 'base.pokemon_type_train="dragon"'`).
3. Compare the experiments with `dvc exp show`.

If you want to move beyond the command line for your experiments, take a look at
[the DVC extension for Visual Studio
Code](https://marketplace.visualstudio.com/items?itemName=Iterative.dvc).

## Workshop - part 2

Now that we can run experiments with our pipeline, let's take our model training
to the cloud! For this second part, we'll be using [CML](https://cml.dev), which
utilizes GitHub Actions (GitLab and Bitbucket equivalents also work).

1. Navigate to your repository on GitHub and enable Actions from the settings.
2. Create a `.github/workflows` directory in your project root.
3. Create a `workflow.yaml` in the newly created directory and start with a
   basic template:

   ```yaml
   name: CML
   on: [push, workflow_dispatch]
   jobs:
   train-and-report:
      runs-on: ubuntu-latest
      container: docker://ghcr.io/iterative/cml:0-dvc2-base1
      steps:
         - uses: actions/checkout@v3
         - run: |
            echo "The workflow is working!"

   ```

4. Add the environment variables CML will need to retrieve the datasets from
   your DVC remote to your GitHub secrets.
5. Adapt the workflow to run `dvc repo` and publish the results as a PR. [See an
   example
   here](https://github.com/iterative/cml_dvc_case/blob/master/.github/workflows/cml.yaml).

