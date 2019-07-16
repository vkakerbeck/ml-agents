# # Unity ML-Agents Toolkit
# ## ML-Agent Learning
"""Launches trainers for each External Brains in a Unity Environment."""

import os
import logging
import shutil
import sys
from typing import *
import random
import pandas as pd

import numpy as np
import tensorflow as tf
from time import time

from mlagentsdev.envs import AllBrainInfo, BrainParameters
from mlagentsdev.envs.base_unity_environment import BaseUnityEnvironment
from mlagentsdev.envs.exception import UnityEnvironmentException
from mlagentsdev.trainers import Trainer
from mlagentsdev.trainers.ppo.trainer import PPOTrainer
from mlagentsdev.trainers.bc.offline_trainer import OfflineBCTrainer
from mlagentsdev.trainers.bc.online_trainer import OnlineBCTrainer
from mlagentsdev.trainers.meta_curriculum import MetaCurriculum
#from mlagentsdev.trainers.depth_extractor import Depth_Extractor


class TrainerController(object):
    def __init__(self,
                 model_path: str,
                 summaries_dir: str,
                 run_id: str,
                 save_freq: int,
                 meta_curriculum: Optional[MetaCurriculum],
                 load: bool,
                 train: bool,
                 keep_checkpoints: int,
                 lesson: Optional[int],
                 external_brains: Dict[str, BrainParameters],
                 training_seed: int,
                 fast_simulation: bool,
                 save_obs: bool,
                 num_envs: int,
                 seed_curriculum: int,
                 use_depth: bool):
        """
        :param model_path: Path to save the model.
        :param summaries_dir: Folder to save training summaries.
        :param run_id: The sub-directory name for model and summary statistics
        :param save_freq: Frequency at which to save model
        :param meta_curriculum: MetaCurriculum object which stores information about all curricula.
        :param load: Whether to load the model or randomly initialize.
        :param train: Whether to train model, or only run inference.
        :param keep_checkpoints: How many model checkpoints to keep.
        :param lesson: Start learning from this lesson.
        :param external_brains: dictionary of external brain names to BrainInfo objects.
        :param training_seed: Seed to use for Numpy and Tensorflow random number generation.
        :param save_obs: Whether to save observations of good runs.
        :param num_envs: Number of parallel environments.
        :param seed_curriculum: Whether to use curriculum learning by showing easy seeds first.
        :param use_depth: Augment visual information with depth information.
        """

        self.model_path = model_path
        self.summaries_dir = summaries_dir
        self.external_brains = external_brains
        self.external_brain_names = external_brains.keys()
        self.logger = logging.getLogger('mlagentsdev.envs')
        self.run_id = run_id
        self.save_freq = save_freq
        self.lesson = lesson
        self.load_model = load
        self.train_model = train
        self.keep_checkpoints = keep_checkpoints
        self.trainers: Dict[str, Trainer] = {}
        self.trainer_metrics: Dict[str, TrainerMetrics] = {}
        self.global_step = 0
        self.meta_curriculum = meta_curriculum
        self.seed = training_seed
        self.training_start_time = time()
        self.fast_simulation = fast_simulation
        self.save_obs = save_obs
        self.num_envs = num_envs
        self.seed_logger = logging.getLogger('seed_logger')
        self.seed_curriculum = seed_curriculum
        self.use_depth = use_depth
        self.depth_extractor = []
        if self.seed_curriculum:
            self.seed_difficulties = []
            self.seed_lesson = 1
        np.random.seed(self.seed)
        tf.set_random_seed(self.seed)

    def _get_measure_vals(self):
        if self.meta_curriculum:
            brain_names_to_measure_vals = {}
            for brain_name, curriculum \
                in self.meta_curriculum.brains_to_curriculums.items():
                if curriculum.measure == 'progress':
                    measure_val = (self.trainers[brain_name].get_step /
                        self.trainers[brain_name].get_max_steps)
                    brain_names_to_measure_vals[brain_name] = measure_val
                elif curriculum.measure == 'reward':
                    measure_val = np.mean(self.trainers[brain_name]
                                          .reward_buffer)
                    brain_names_to_measure_vals[brain_name] = measure_val
            return brain_names_to_measure_vals
        else:
            return None

    def _save_model(self, steps=0):
        """
        Saves current model to checkpoint folder.
        :param steps: Current number of steps in training process.
        :param saver: Tensorflow saver for session.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].save_model()
        self.logger.info('Saved Model')

    def _save_model_when_interrupted(self, steps=0):
        self.logger.info('Learning was interrupted. Please wait '
                         'while the graph is generated.')
        self._save_model(steps)

    def _write_training_metrics(self):
        """
        Write all CSV metrics
        :return:
        """
        for brain_name in self.trainers.keys():
            if brain_name in self.trainer_metrics:
                self.trainers[brain_name].write_training_metrics()

    def _export_graph(self):
        """
        Exports latest saved models to .nn format for Unity embedding.
        """
        for brain_name in self.trainers.keys():
            self.trainers[brain_name].export_model()

    def initialize_trainers(self, trainer_config: Dict[str, Dict[str, str]]):
        """
        Initialization of the trainers
        :param trainer_config: The configurations of the trainers
        """
        trainer_parameters_dict = {}
        for brain_name in self.external_brains:
            trainer_parameters = trainer_config['default'].copy()
            trainer_parameters['summary_path'] = '{basedir}/{name}'.format(
                basedir=self.summaries_dir,
                name=str(self.run_id) + '_' + brain_name)
            trainer_parameters['model_path'] = '{basedir}/{name}'.format(
                basedir=self.model_path,
                name=brain_name)
            trainer_parameters['keep_checkpoints'] = self.keep_checkpoints
            if brain_name in trainer_config:
                _brain_key = brain_name
                while not isinstance(trainer_config[_brain_key], dict):
                    _brain_key = trainer_config[_brain_key]
                for k in trainer_config[_brain_key]:
                    trainer_parameters[k] = trainer_config[_brain_key][k]
            trainer_parameters_dict[brain_name] = trainer_parameters.copy()
        for brain_name in self.external_brains:
            if trainer_parameters_dict[brain_name]['trainer'] == 'offline_bc':
                self.trainers[brain_name] = OfflineBCTrainer(
                    self.external_brains[brain_name],
                    trainer_parameters_dict[brain_name], self.train_model,
                    self.load_model, self.seed, self.run_id)
            elif trainer_parameters_dict[brain_name]['trainer'] == 'online_bc':
                self.trainers[brain_name] = OnlineBCTrainer(
                    self.external_brains[brain_name],
                    trainer_parameters_dict[brain_name], self.train_model,
                    self.load_model, self.seed, self.run_id)
            elif trainer_parameters_dict[brain_name]['trainer'] == 'ppo':
                self.trainers[brain_name] = PPOTrainer(
                    self.external_brains[brain_name],
                    self.meta_curriculum
                        .brains_to_curriculums[brain_name]
                        .min_lesson_length if self.meta_curriculum else 0,
                    trainer_parameters_dict[brain_name],
                    self.train_model, self.load_model, self.seed,
                    self.run_id,self.save_obs,self.num_envs, self.use_depth)
                self.trainer_metrics[brain_name] = self.trainers[brain_name].trainer_metrics
            else:
                raise UnityEnvironmentException('The trainer config contains '
                                                'an unknown trainer type for '
                                                'brain {}'
                                                .format(brain_name))

    @staticmethod
    def _create_model_path(model_path):
        try:
            if not os.path.exists(model_path):
                os.makedirs(model_path)
        except Exception:
            raise UnityEnvironmentException('The folder {} containing the '
                                            'generated model could not be '
                                            'accessed. Please make sure the '
                                            'permissions are set correctly.'
                                            .format(model_path))

    def _reset_env(self, env: BaseUnityEnvironment, config=None):
        """Resets the environment.

        Returns:
            A Data structure corresponding to the initial reset state of the
            environment.
        """
        #XX Adjust to be more uniform
        if self.meta_curriculum is not None:
            return env.reset(train_mode=self.fast_simulation, config=self.meta_curriculum.get_config())
        else:
            if config==None:
                return env.reset(train_mode=self.fast_simulation)
            else:
                return env.reset_one(train_mode=self.fast_simulation, config = config)

    def set_up_logger(self):
        f_handler = logging.FileHandler('./seed_stats/'+str(self.run_id) + '_' + 'SeedStats.csv')
        f_handler.setLevel(logging.INFO)
        f_format = logging.Formatter('%(relativeCreated)6d,%(message)s')
        f_handler.setFormatter(f_format)
        self.seed_logger.addHandler(f_handler)

    def getSeedCurriculum(self):
        seed_stats = pd.read_csv('./seed_stats/TowerF4_BaselineCopy-0_SeedStatsF.csv')#XX make universal
        seed_stats = seed_stats[1:]
        seed_stats['Tower Seed'] = seed_stats['Tower Seed'].astype(int)
        seed_stats_g = seed_stats.groupby(['Tower Seed']).mean()
        seed_stats_gs = seed_stats_g.sort_values('Reward')
        seedlist = seed_stats_gs.index
        return seedlist

    def getCurriculumSeed(self):
        if self.seed_lesson == 0:
            seed = self.seed_difficulties[random.randint(61, 100)]#easiest seeds
        elif self.seed_lesson == 1:
            seed = self.seed_difficulties[random.randint(31, 61)]#medium difficulty
        else:
            seed = self.seed_difficulties[random.randint(0, 31)]#hardest seeds
        return seed

    def start_learning(self, env: BaseUnityEnvironment, trainer_config):
        # TODO: Should be able to start learning at different lesson numbers
        # for each curriculum.
        if self.meta_curriculum is not None:
            self.meta_curriculum.set_all_curriculums_to_lesson_num(self.lesson)
        self._create_model_path(self.model_path)

        tf.reset_default_graph()

        # Prevent a single session from taking all GPU memory.
        self.initialize_trainers(trainer_config)
        for _, t in self.trainers.items():
            self.logger.info(t)

        if self.train_model:
            for brain_name, trainer in self.trainers.items():
                trainer.write_tensorboard_text('Hyperparameters',
                                               trainer.parameters)
        try:
            curr_info = self._reset_env(env)
            if self.use_depth:
                self.depth_extractor = Depth_Extractor('C:/Users/vivia/Desktop/ml-agents-dev/depth_models/')#XX make universal
                for e in range(self.num_envs):
                    cur_vis = curr_info['LearningBrain'].visual_observations[0][e]
                    d = self.depth_extractor.compute_depths(cur_vis,'crop')
                    curr_info['LearningBrain'].visual_observations[0][e] = np.concatenate((cur_vis,d),axis=2)
                    print(curr_info['LearningBrain'].visual_observations[0][e].shape)
            if self.train_model:
                self.set_up_logger()
                self.seed_difficulties = self.getSeedCurriculum()
                header = 'Agent,Tower Seed,Reward,Floor,Episode Length,Keys'#XX don't add header if file already exists
                self.seed_logger.info(header)
                startS = 0
                print(self.seed_lesson)
            while any([t.get_step <= t.get_max_steps \
                       for k, t in self.trainers.items()]) \
                  or not self.train_model:#XX first seed not random
                new_info = self.take_step(env, curr_info)
                info = new_info['LearningBrain']
                if (self.train_model and (startS == 0 and t.get_step>0)):
                    startS = t.get_step
                    print('startS: ' + str(startS))
                if info.local_done[0]:#XX Make possible to record all agents
                    stats = self.trainers['LearningBrain'].getStats()
                    """info_str = ('Agent ' + str(stats[0]) + ': Keys: ' + str(stats[1]) + ' Floor: ' + str(stats[2]) +
                        ' Episode Length: ' + str(stats[3]) + ' Reward: ' + str(stats[4]) + ' Tower Seed: ' +
                        str(env.reset_parameters['tower-seed']))
                    print(info_str)"""
                    if self.train_model:
                        info_log = (str(stats[0]) + ',' + str(env.reset_parameters['tower-seed']) + ',' + str(stats[4]) +
                        ',' + str(stats[2]) + ',' + str(stats[3]) + ',' + str(stats[1]))
                        self.seed_logger.info(info_log)
                        if (t.get_step - startS > 5000000) and self.seed_curriculum:#XX add incrementation dependent on average reward
                            self.seed_lesson = self.seed_lesson + 1#XX tensorflow summaries - add seed lesson
                            startS = t.get_step
                            print("incrementing seed lesson - now in lesson "+str(self.seed_lesson))
                    if self.seed_curriculum:
                        seed = int(self.getCurriculumSeed())
                    else:
                        seed = random.randint(0, 100)
                    curr_info = self._reset_env(env, config = {'tower-seed': seed})

                self.global_step += 1
                if self.global_step % self.save_freq == 0 and self.global_step != 0 \
                        and self.train_model:
                    # Save Tensorflow model
                    self._save_model(steps=self.global_step)
                curr_info = new_info
            # Final save Tensorflow model
            if self.global_step != 0 and self.train_model:
                self._save_model(steps=self.global_step)
        except KeyboardInterrupt:
            if self.train_model:
                self._save_model_when_interrupted(steps=self.global_step)
            pass
        env.close()
        if self.train_model:
            self._write_training_metrics()
            self._export_graph()

    def take_step(self, env: BaseUnityEnvironment, curr_info: AllBrainInfo):
        if self.meta_curriculum:
            # Get the sizes of the reward buffers.
            reward_buff_sizes = {k: len(t.reward_buffer)
                                 for (k, t) in self.trainers.items()}
            # Attempt to increment the lessons of the brains who
            # were ready.
            lessons_incremented = \
                self.meta_curriculum.increment_lessons(
                    self._get_measure_vals(),
                    reward_buff_sizes=reward_buff_sizes)
        else:
            lessons_incremented = {}

        # If any lessons were incremented or the environment is
        # ready to be reset
        if (self.meta_curriculum
                and any(lessons_incremented.values())):
            curr_info = self._reset_env(env)
            for brain_name, trainer in self.trainers.items():
                trainer.end_episode()
            for brain_name, changed in lessons_incremented.items():
                if changed:
                    self.trainers[brain_name].reward_buffer.clear()
        """elif env.global_done:
            print("global done")
            curr_info = self._reset_env(env)
            for brain_name, trainer in self.trainers.items():
                trainer.end_episode()"""

        # Decide and take an action
        take_action_vector = {}
        take_action_memories = {}
        take_action_text = {}
        take_action_value = {}
        take_action_outputs = {}
        for brain_name, trainer in self.trainers.items():
            action_info = trainer.get_action(curr_info[brain_name])
            take_action_vector[brain_name] = action_info.action
            take_action_memories[brain_name] = action_info.memory
            take_action_text[brain_name] = action_info.text
            take_action_value[brain_name] = action_info.value
            take_action_outputs[brain_name] = action_info.outputs
        time_start_step = time()

        new_info = env.step(
            vector_action=take_action_vector,
            memory=take_action_memories,
            text_action=take_action_text,
            value=take_action_value
        )

        if self.use_depth:
            for e in range(self.num_envs):
                vis = new_info['LearningBrain'].visual_observations[0][e]
                d = self.depth_extractor.compute_depths(vis,'crop')
                new_info['LearningBrain'].visual_observations[0][e] = np.concatenate((vis,d),axis=2)

        delta_time_step = time() - time_start_step
        for brain_name, trainer in self.trainers.items():
            if brain_name in self.trainer_metrics:
                self.trainer_metrics[brain_name].add_delta_step(delta_time_step)
            trainer.add_experiences(curr_info, new_info,
                                    take_action_outputs[brain_name])
            trainer.process_experiences(curr_info, new_info)
            if trainer.is_ready_update() and self.train_model \
                    and trainer.get_step <= trainer.get_max_steps:
                # Perform gradient descent with experience buffer

                trainer.update_policy()
            # Write training statistics to Tensorboard.
            delta_train_start = time() - self.training_start_time
            if self.meta_curriculum is not None:
                trainer.write_summary(
                    self.global_step,
                    delta_train_start, lesson_num=self.meta_curriculum
                        .brains_to_curriculums[brain_name]
                        .lesson_num)
            else:
                trainer.write_summary(self.global_step, delta_train_start)
            if self.train_model \
                    and trainer.get_step <= trainer.get_max_steps:
                trainer.increment_step_and_update_last_reward()
        return new_info
