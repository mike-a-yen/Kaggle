from pathlib import Path

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
    
    lm_data = trainer.build_databunch(split_ds.extra_df, split_ds.trainval_df)
    trainer.set_lm_learner(lm_data)
    trainer.fit()
    
    lm_fine_data = trainer.build_databunch(split_ds.trainval_df, split_ds.test_df)
    trainer.set_lm_learner(lm_fine_data)
    trainer.learner.load_encoder('ft_enc')
    trainer.learner.unfreeze()
    lr = slice(trainer.training_params['lr'] / 100, trainer.training_params['lr'])
    trainer.fit(lr=lr)

    data = trainer.build_clas_databunch(split_ds.train_df, split_ds.val_df, split_ds.test_df)
    trainer.set_clas_learner(data)

    trainer.learner.fit(4, slice(1e-6, 1e-2))
    trainer.learner.unfreeze()
    trainer.learner.fit(4, slice(1e-6, 3e-3))
    trainer.learner.save('clas_model')
    trainer.set_predictions(split_ds, skip=['extra_df'])
    trainer.save_data(split_ds)


if __name__ == '__main__':
    import logging
    import fire
    logging.basicConfig(level=logging.INFO)
    fire.Fire(main)
