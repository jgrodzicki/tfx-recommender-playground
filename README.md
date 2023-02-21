# TFX Recommender Playground

This repo contains the code which runs the TFX pipeline. By default, the data is downloaded using Kaggle API (also 
locally stored sample data can be used) and creates a retrieval recommender model which given
`user_id` predicts the recipes he/she might've tried and rated - 
[kaggle_dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

Note: The code was developed to contain knowledge about tfx and options of running it. It wasn't developed to be 
a generic multipurpose tool.

## Building
```bash
docker build . -t tfx-recommender-playground
```

## Running

### Run locally
```bash
docker run tfx-recommender-playground
```

By default, flags defined in the `CMD` in the `Dockerfile` will be used. To use different set of flags just pass them 
in the command, e.g.
```bash
docker run tfx-recommender-playground --use-local-sample-data --runner-env=local --epochs=5
```

### Run on Vertex AI
1. Push the docker to the Container Registry / Artifact Registry
2. Create a new training job in Vertex AI
   1. Getting the dataset is handled in the code, hence leave `Dataset - No managed dataset` as the default
   2. Leave 2nd option as `Train a new model`, possibly a service account has to be created - once it's done and 
      permissions are set, select it from the advanced options.
   3. Custom training container should be used, browse in the registry to select the one you pushed. Arguments should 
      also be provided. Minimal set is `--use-local-sample-data` & `--runner-env=vertex_ai`
   4. Select the machine type you desire
   5. After finished setup, the job will be started.

Started job will run the argument parsing and the setup leading to creating and running a pipeline. It's the way
that `KubeflowV2DagRunner` runs stuff. In order to have access to all the custom code created, it will use the docker 
image defined in the code in `create_kubeflow_dag_runner` function in `runner_factory.py` file. To test the 
solution end-to-end, tag defined in the code should correspond to the newly pushed image