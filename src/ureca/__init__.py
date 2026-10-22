"""PD classification from raw sensor waveforms.

The package follows the pipeline; each stage only uses the ones above it.

    config.py, db.py   shared: where the data lives, the SQLite database
    ingest/            1. raw .txt batches -> validated files + database rows
        pipeline.py       validate, copy, register, reject (`python -m ureca.ingest`)
        labels.py         class folder name -> class label
    dataset/           2. database -> model-ready data
        folds.py          recording sessions and cross-validation folds
        loading.py        file index, waveform cache, fold_split
        torch_data.py     preprocessing, PyTorch Dataset, samplers
    models/            3. waveforms -> predictions
        features.py       hand-made features (baseline input)
        cnn.py            1-D CNN
    evaluate.py        4. out-of-fold predictions and scores
    plotting.py        plot style for the notebooks
"""
