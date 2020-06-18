# Copyright (c) 2020, Ioana Bica
# Modified 2020, Chris Lucas

import os
import argparse
import logging

from pathlib import Path

# from CRN_encoder_evaluate import test_CRN_encoder
# from CRN_decoder_evaluate import test_CRN_decoder
from dataset.cancer_loader import CancerPatients
from dataset.cancer_simulation import get_cancer_sim_data
from utils import ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--chemo_coeff",
        default=2,
        type=int,
        help="Degree of time-dependent confounding for chemotherapy",
    )
    parser.add_argument(
        "--radio_coeff",
        default=2,
        type=int,
        help="Degree of time-dependent confounding for radiotherapy",
    )
    parser.add_argument(
        "--results_dir",
        default=Path("results"),
        type=Path,
        help="Results output directory",
    )
    parser.add_argument("--model_name", type=str, default="crn_test_2")
    parser.add_argument("--b_encoder_hyperparm_tuning", type=bool, default=False)
    parser.add_argument("--b_decoder_hyperparm_tuning", type=bool, default=False)
    parser.add_argument(
        "--num_patients", type=int, default=1000, help="Number of patients to simulate"
    )
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()

    ensure_dir(args.results_dir)

    logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

    pickle_map = get_cancer_sim_data(
        chemo_coeff=args.chemo_coeff,
        radio_coeff=args.radio_coeff,
        b_load=True,
        b_save=True,
        num_patients=args.num_patients,
    )

    train_loader = CancerPatients(
        data=pickle_map["training_data"],
        scaling_data=pickle_map["scaling_data"],
        chemo_coeff=args.chemo_coeff,
        radio_coeff=args.radio_coeff,
        num_time_steps=pickle_map["num_time_steps"],
        window_size=pickle_map["window_size"],
    )

    validation_loader = CancerPatients(
        data=pickle_map["validation_data"],
        scaling_data=pickle_map["scaling_data"],
        chemo_coeff=args.chemo_coeff,
        radio_coeff=args.radio_coeff,
        num_time_steps=pickle_map["num_time_steps"],
        window_size=pickle_map["window_size"],
    )

    test_loader = CancerPatients(
        data=pickle_map["test_data"],
        scaling_data=pickle_map["scaling_data"],
        chemo_coeff=args.chemo_coeff,
        radio_coeff=args.radio_coeff,
        num_time_steps=pickle_map["num_time_steps"],
        window_size=pickle_map["window_size"],
    )

    import pdb

    pdb.set_trace()

    # encoder_model_name = 'encoder_' + args.model_name
    # encoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, encoder_model_name)

    # models_dir = args.results_dir / "crn_models"
    # ensure_dir(models_dir)

    # rmse_encoder = test_CRN_encoder(pickle_map=pickle_map, models_dir=models_dir,
    #                                 encoder_model_name=encoder_model_name,
    #                                 encoder_hyperparams_file=encoder_hyperparams_file,
    #                                 b_encoder_hyperparm_tuning=args.b_encoder_hyperparm_tuning)

    # decoder_model_name = 'decoder_' + args.model_name
    # decoder_hyperparams_file = '{}/{}_best_hyperparams.txt'.format(args.results_dir, decoder_model_name)

    # """
    # The counterfactual test data for a sequence of treatments in the future was simulated for a
    # projection horizon of 5 timesteps.

    # """

    # max_projection_horizon = 5
    # projection_horizon = 5

    # rmse_decoder = test_CRN_decoder(pickle_map=pickle_map, max_projection_horizon=max_projection_horizon,
    #                                 projection_horizon=projection_horizon,
    #                                 models_dir=models_dir,
    #                                 encoder_model_name=encoder_model_name,
    #                                 encoder_hyperparams_file=encoder_hyperparams_file,
    #                                 decoder_model_name=decoder_model_name,
    #                                 decoder_hyperparams_file=decoder_hyperparams_file,
    #                                 b_decoder_hyperparm_tuning=args.b_decoder_hyperparm_tuning)

    # logging.info("Chemo coeff {} | Radio coeff {}".format(args.chemo_coeff, args.radio_coeff))
    # print("RMSE for one-step-ahead prediction.")
    # print(rmse_encoder)

    # print("Results for 5-step-ahead prediction.")
    # print(rmse_decoder)
