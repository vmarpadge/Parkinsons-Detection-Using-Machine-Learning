# Contributing

1. Fork the repo and create a branch from `master`.
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python train.py`
4. Test your changes: `python predict.py <your_file.wav>`
5. Open a pull request.

Areas where help is especially welcome:
- Unit tests for `extract_features()`, `train()`, and `predict()`
- Improving model accuracy (currently ~65%)
- Supporting additional audio formats
