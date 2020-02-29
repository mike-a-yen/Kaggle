from pathlib import Path

import pandas as pd
import toml

from src.trainer import Trainer


_PROJECT_DIR = Path(__file__).parents[1].resolve()
_CONFIG_DIR = _PROJECT_DIR / 'src' / 'config'


def load_config(config_file: str) -> dict:
    with open(_CONFIG_DIR / config_file) as fo:
        return toml.load(fo)


def main(
        model_config: str = 'model.toml',
        training_config: str = 'training.toml',
        subsample: int = 0,
        match: float = 0.0,
        pretrain: bool = True,
        sync: bool = False,
        run_id: str = None,
        with_pos_embedding: bool = False,
        **kwargs
) -> None:
    model_params = load_config(model_config)
    training_params = load_config(training_config)
    training_params.update(kwargs)

    trainer = Trainer(model_params, training_params, id=run_id, sync=sync)

    split_ds = trainer.load_data(subsample, match_thresh=match)

    trainer.analyze(split_ds)
    trainer.set_datapipeline(split_ds)
    trainer.data_pipeline.display_vocab_coverage()

    if pretrain:
        lm_data = trainer.build_databunch(split_ds.extra_df, split_ds.trainval_df)
        trainer.set_lm_learner(lm_data)
        trainer.fit()

        lm_fine_data = trainer.build_databunch(split_ds.trainval_df, split_ds.test_df)
        trainer.set_lm_learner(lm_fine_data)
        trainer.learner.load_encoder('ft_enc')
        trainer.fit()

    if 'annotation' in split_ds.extra_df.columns and split_ds.target_col == 'annotation':
        labeled_extra_data = split_ds.extra_df[~split_ds.extra_df.annotation.isnull()]
        clas_train_df = pd.concat([split_ds.train_df, labeled_extra_data], sort=False, ignore_index=True)
    else:
        clas_train_df = split_ds.train_df.copy()

    data = trainer.build_clas_databunch(clas_train_df, split_ds.val_df, split_ds.test_df)
    trainer.set_clas_learner(data)

    if pretrain:
        trainer.learner.fit(4, slice(1e-6, training_params['lr']))
        trainer.learner.unfreeze()
        trainer.learner.fit(4, slice(1e-6, training_params['lr']))
    else:
        trainer.learner.unfreeze()
        trainer.learner.fit(epochs=training_params['epochs'], lr=training_params['lr'])
    trainer.learner.save('clas_model')
    trainer.learner.export('clas_learner.pkl')
    trainer.set_predictions(split_ds)
    trainer.save_data(split_ds)


if __name__ == '__main__':
    import logging
    import fire
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
