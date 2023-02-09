# TFX Recommender Playground

This repo contains the code which runs the TFX pipeline. By default the data is downloaded using Kaggle API (also 
locally stored sample data can be used) and creates a retrieval recommender model which given
`user_id` predicts the recipes he/she might've tried and rated - 
[kaggle_dataset](https://www.kaggle.com/datasets/shuyangli94/food-com-recipes-and-user-interactions)

### Running
1. Build the docker:
```bash
docker build . -t tfx-recommender-playground
```

2. Run the docker interactively
```bash
docker run -it tfx-recommender-playground bash
```

3. Run main - you can provide arguments defined in the `parser.py` file
```bash
python main.py
```
To run without Kaggle credentials, `--use-local-sample-data` flag should be used.