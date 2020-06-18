from collections import defaultdict

import pickle

import numpy as np

from torch.utils.data import Dataset


class CancerPatients(Dataset):
    def __init__(
        self,
        data,
        scaling_data,
        chemo_coeff,
        radio_coeff,
        num_time_steps,
        window_size,
        factuals=None,
        transform=None,
    ):
        self._data = defaultdict(dict)

        self._process_input_data(
            input_data=data, scaling_data=scaling_data,
        )

        self._return_factual_data = False

        if factuals is not None:
            self._process_input_data(
                input_data=factuals, scaling_data=scaling_data, data_key="factuals",
            )
            self._return_factual_data = True

        self._chemo_coeff = chemo_coeff
        self._radio_coeff = radio_coeff
        self._num_time_steps = num_time_steps
        self._window_size = window_size

        self.transform = transform

        self._data_keys = [f"_{_key}" for _key in data.keys()]

    def return_factual_data(self, flag=True):
        self._return_factual_data = flag

    def _process_input_data(self, input_data, scaling_data, data_key="default"):
        offset = 1
        horizon = 1

        mean, std = scaling_data

        mean["chemo_application"] = 0
        mean["radio_application"] = 0
        std["chemo_application"] = 1
        std["radio_application"] = 1

        input_means = mean[
            ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
        ].values.flatten()
        input_stds = std[
            ["cancer_volume", "patient_types", "chemo_application", "radio_application"]
        ].values.flatten()

        # Continuous values
        cancer_volume = (input_data["cancer_volume"] - mean["cancer_volume"]) / std[
            "cancer_volume"
        ]
        patient_types = (input_data["patient_types"] - mean["patient_types"]) / std[
            "patient_types"
        ]

        patient_types = np.stack(
            [patient_types for t in range(cancer_volume.shape[1])], axis=1
        )

        # Binary application
        chemo_application = input_data["chemo_application"]
        radio_application = input_data["radio_application"]
        sequence_lengths = input_data["sequence_lengths"]

        # Convert treatments to one-hot encoding
        treatments = np.concatenate(
            [
                chemo_application[:, :-offset, np.newaxis],
                radio_application[:, :-offset, np.newaxis],
            ],
            axis=-1,
        )

        one_hot_treatments = np.zeros(
            shape=(treatments.shape[0], treatments.shape[1], 4)
        )
        for patient_id in range(treatments.shape[0]):
            for timestep in range(treatments.shape[1]):
                if (
                    treatments[patient_id][timestep][0] == 0
                    and treatments[patient_id][timestep][1] == 0
                ):
                    one_hot_treatments[patient_id][timestep] = [1, 0, 0, 0]
                elif (
                    treatments[patient_id][timestep][0] == 1
                    and treatments[patient_id][timestep][1] == 0
                ):
                    one_hot_treatments[patient_id][timestep] = [0, 1, 0, 0]
                elif (
                    treatments[patient_id][timestep][0] == 0
                    and treatments[patient_id][timestep][1] == 1
                ):
                    one_hot_treatments[patient_id][timestep] = [0, 0, 1, 0]
                elif (
                    treatments[patient_id][timestep][0] == 1
                    and treatments[patient_id][timestep][1] == 1
                ):
                    one_hot_treatments[patient_id][timestep] = [0, 0, 0, 1]

        one_hot_previous_treatments = one_hot_treatments[:, :-1, :]

        current_covariates = np.concatenate(
            [
                cancer_volume[:, :-offset, np.newaxis],
                patient_types[:, :-offset, np.newaxis],
            ],
            axis=-1,
        )
        outputs = cancer_volume[:, horizon:, np.newaxis]

        output_means = mean[["cancer_volume"]].values.flatten()[
            0
        ]  # because we only need scalars here
        output_stds = std[["cancer_volume"]].values.flatten()[0]

        # Add active entires
        active_entries = np.zeros(outputs.shape)

        for i in range(sequence_lengths.shape[0]):
            sequence_length = int(sequence_lengths[i])
            active_entries[i, :sequence_length, :] = 1

        self._data[data_key]["current_covariates"] = current_covariates
        self._data[data_key]["previous_treatments"] = one_hot_previous_treatments
        self._data[data_key]["current_treatments"] = one_hot_treatments
        self._data[data_key]["outputs"] = outputs
        self._data[data_key]["active_entries"] = active_entries

        self._data[data_key]["unscaled_outputs"] = (
            outputs * std["cancer_volume"] + mean["cancer_volume"]
        )
        self._data[data_key]["input_means"] = input_means
        self._data[data_key]["inputs_stds"] = input_stds
        self._data[data_key]["output_means"] = output_means
        self._data[data_key]["output_stds"] = output_stds

        # this is placeholder for some RNN decoder input
        self._data[data_key]["init_state"] = np.zeros_like(outputs)

    def __len__(self):
        return self._data["default"]["current_covariates"].shape[0]

    def __getitem__(self, idx):
        output_keys = [
            "current_covariates",
            "previous_treatments",
            "current_treatments",
            "outputs",
            "active_entries",
            "init_state",
        ]
        sample = [self._data["default"][key][idx] for key in output_keys]

        if not self._return_factual_data:
            return sample
        else:
            factual_sample = [self._data["factuals"][key][idx] for key in output_keys]
            return sample, factual_sample


def sum_all(inp):
    summers = []
    for thing in inp:
        summers.append(thing.sum())
    print(summers)


if __name__ == "__main__":
    data = CancerPatients(filename="/data/chrisl/CRN-data/training.p")
    sample = data[0]
    print("Training sample:")
    sum_all(sample)

    test_data = CancerPatients(filename="/data/chrisl/CRN-data/test.p")
    tample = test_data[0]
    print("Test sample:")
    print(len(tample))
    sum_all(tample[0])
    sum_all(tample[1])

    test_data.return_factual_data(False)
    ntample = test_data[0]
    print("Test factual sample:")
    print(len(ntample))
    sum_all(ntample)

    # import pdb

    # pdb.set_trace()
