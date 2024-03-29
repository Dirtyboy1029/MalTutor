import os.path as path
import time

import tensorflow as tf
import numpy as np

from core.ensemble.vanilla import Vanilla
from core.ensemble.model_hp import train_hparam, anchor_hparam
from core.ensemble.model_lib import model_builder
from ..tools import utils
from ..config import logging

logger = logging.getLogger('ensemble.vanilla')


class AnchorEnsemble(Vanilla):
    def __init__(self,
                 architecture_type='dnn',
                 base_model=None,
                 n_members=2,
                 model_directory=None,
                 name='ANCHOR'):
        """
        initialization
        :param architecture_type: the type of base model
        :param base_model: an object of base model
        :param n_members: number of base models
        :param model_directory: a folder for saving ensemble weights
        """
        super(AnchorEnsemble, self).__init__(architecture_type, base_model, n_members, model_directory)
        self.hparam = utils.merge_namedtuples(train_hparam, anchor_hparam)
        self.ensemble_type = 'anchor'
        self.name = name.lower()
        self.save_dir = path.join(self.model_directory, self.name)

    def build_model(self, input_dim=None):
        """
        Build an ensemble model -- only the homogeneous structure is considered
        :param input_dim: integer or list, input dimension shall be set in some cases under eager mode
        """
        callable_graph = model_builder(self.architecture_type)

        @callable_graph(input_dim)
        def _builder():
            seed = np.random.choice(self.hparam.random_seed)
            return utils.produce_layer(self.ensemble_type,
                                       scale=self.hparam.scale,
                                       batch_size=self.hparam.batch_size,
                                       seed=seed)

        self.base_model = _builder()

    def model_generator(self):
        try:
            for m in range(self.n_members):
                self.base_model = None
                self.load_ensemble_weights(m)
                yield self.base_model
        except Exception as e:
            raise Exception("Cannot load model:{}.".format(str(e)))

    def fit(self, train_set, validation_set=None, input_dim=None, **kwargs):
        """
        fit the ensemble by producing a lists of model weights
        :param train_set: tf.data.Dataset, the type shall accommodate to the input format of Tensorflow models
        :param validation_set: validation data, optional
        :param input_dim: integer or list, input dimension except for the batch size
        """
        # training
        logger.info("hyper-parameters:")
        logger.info(dict(self.hparam._asdict()))
        logger.info("...training start!")
        np.random.seed(self.hparam.random_seed)
        train_set = train_set.shuffle(buffer_size=100, reshuffle_each_iteration=True)
        for member_idx in range(self.n_members):
            self.base_model = None
            self.build_model(input_dim=input_dim)

            self.base_model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=self.hparam.learning_rate),
                loss=tf.keras.losses.BinaryCrossentropy(),
                metrics=[tf.keras.metrics.BinaryAccuracy()],
            )
            for epoch in range(self.hparam.n_epochs):
                total_time = 0.
                msg = 'Epoch {}/{}, and member {}/{}'.format(epoch + 1,
                                                             self.hparam.n_epochs, member_idx + 1,
                                                             self.n_members)
                print(msg)
                start_time = time.time()
                self.base_model.fit(train_set,
                                    epochs=epoch + 1,
                                    initial_epoch=epoch,
                                    validation_data=validation_set
                                    )
                end_time = time.time()
                total_time += end_time - start_time
                # saving
                logger.info('Training ensemble costs {} seconds at this epoch'.format(total_time))
                if (epoch + 1) % self.hparam.interval == 0:
                    self.save_ensemble_weights(member_idx)

    def save_ensemble_weights(self, member_idx=0):
        if not path.exists(path.join(self.save_dir, self.architecture_type + '_{}'.format(member_idx))):
            utils.mkdir(path.join(self.save_dir, self.architecture_type + '_{}'.format(member_idx)))
        # save model configuration
        self.base_model.save(path.join(self.save_dir, self.architecture_type + '_{}'.format(member_idx)))
        print("Save the model to directory {}".format(self.save_dir))

    def load_ensemble_weights(self, member_idx=0):
        if path.exists(path.join(self.save_dir, self.architecture_type + '_{}'.format(member_idx))):
            self.base_model = tf.keras.models.load_model(
                path.join(self.save_dir, self.architecture_type + '_{}'.format(member_idx)))

    def get_n_members(self):
        return self.n_members

    def update_weights(self, member_idx, model_weights, optimizer_weights=None):
        raise NotImplementedError
