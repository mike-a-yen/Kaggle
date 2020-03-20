from pathlib import Path

import pandas as pd
import toml
import wandb

from bot.db.utils import log_model, log_evaluation
from nn_toolkit.utils import unfreeze_parameters, count_trainable_params
from src.trainer import Trainer
from src.evaluator import Evaluator


_PROJECT_DIR = Path(__file__).parents[1].resolve()
_CONFIG_DIR = _PROJECT_DIR / 'src' / 'config'


def load_config(config_file: str) -> dict:
    with open(_CONFIG_DIR / config_file) as fo:
        return toml.load(fo)


def main(
        model_config: str = 'model.toml',
        training_config: str = 'training.toml',
        limit: int = None,
        match: float = 0.0,
        pretrain: bool = True,
        sync: bool = False,
        run_id: str = None,
        min_precision: float = 0.99,
        **kwargs
) -> None:
    model_params = load_config(model_config)
    training_params = load_config(training_config)
    training_params.update(kwargs)

    trainer = Trainer(model_params, training_params, id=run_id, sync=sync)

    split_ds = trainer.load_data(limit, match_thresh=match)

    trainer.analyze(split_ds)
    trainer.set_lm_datapipeline(split_ds)
    trainer.set_clas_datapipeline(split_ds)
    trainer.lm_pipeline.display_vocab_coverage()
    trainer.set_lm_model()
    logging.info(f'Trainable params: {count_trainable_params(trainer.lm_model)}')

    lr = training_params['lr']
    if pretrain:
        lm_data = trainer.build_lm_databunch(split_ds)
        trainer.set_lm_learner(lm_data)
        trainer.learner.fit(training_params['epochs'], lr)

    clas_data = trainer.build_clas_databunch(split_ds)
    trainer.set_clas_model()
    trainer.set_clas_learner(clas_data)
    if not pretrain:
        unfreeze_parameters(trainer.clas_model)
    logging.info(f'Trainable params: {count_trainable_params(trainer.clas_model)}')
    trainer.learner.fit(training_params['epochs'], lr)
    if not pretrain:
        unfreeze_parameters(trainer.clas_model)
        logging.info(f'Trainable params: {count_trainable_params(trainer.clas_model)}')
        trainer.learner.fit(2, lr)
    trainer.save_bits()
    trainer.set_predictions(split_ds)
    trainer.save_data(split_ds)

    evaluator = Evaluator(trainer.learner)
    evaluator.evaluate(min_precision=min_precision)
    results = evaluator.get_results()
    trainer.logger.log(results)
    if sync:
        model_id = log_model(wandb.run.id, str(trainer.run_dir))
        log_evaluation(model_id, **results)



if __name__ == '__main__':
    import logging
    import fire
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%d-%b-%y %H:%M:%S'
    )
    fire.Fire(main)
