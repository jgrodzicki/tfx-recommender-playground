# TFX Recommender Playground

### Running
1. Build the docker:
```bash
docker build . -t tfx-recommender-playground
```

2. Run the docker interactively
```bash
docker run -it tfx-recommender-playground
```

3. Run main - you can provide arguments defined in the `parser.py` file
```bash
python main.py
```
To run without Kaggle credentials, `--use-local-sample-data` flag can be used.