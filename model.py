import torch

from pytorch_revgrad import RevGrad


class Flatten(torch.nn.Module):
    def forward(self, x):
        batch_size = x.size()[0]
        # maintain batch_size and flatten other dims
        x = x.view(batch_size, -1)
        return x


class LSTMEmbedder(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_units,
        batchnorm=False,
        drop_prob=0.0,
        activation=torch.nn.ReLU,
    ):
        super(LSTMEmbedder, self).__init__()

        self._lstm_layer = torch.nn.LSTM(
            input_size=input_size, hidden_size=hidden_units, dropout=drop_prob
        )
        self._fc_layer = torch.nn.Linear(hidden_units * input_size, output_size)
        self._activation = activation()
        self._flatten = Flatten()

    def forward(self, input_sequence):
        lstm_output, _ = self._lstm_layer(input_sequence)
        flat_lstm = self._flatten(lstm_output)
        embedding = self._fc_layer(flat_lstm)
        return self._activation(embedding)


class DNN(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_hidden_layers,
        hidden_dim,
        flip_gradient=False,
        batchnorm=False,
        drop_prob=0.0,
        activation=torch.nn.ReLU,
    ):
        super(DNN, self).__init__()

        layers = [
            torch.nn.Linear(input_size, hidden_dim),
            torch.nn.Dropout(drop_prob),
            activation(),
        ]

        if batchnorm:
            raise NotImplementedError

        for i in range(num_hidden_layers):
            layers.append(torch.nn.Linear(hidden_dim, hidden_dim))
            layers.append(torch.nn.Dropout(drop_prob))
            layers.append(activation())

        layers.append(torch.nn.Linear(hidden_dim, output_size))

        if flip_gradient:
            layers.append(RevGrad())

        self._network = torch.nn.Sequential(*layers)

    def forward(self, input_data):
        raw_output = self._network(input_data)
        return raw_output


class CRNModel(torch.nn.Module):
    def __init__(
        self,
        input_size,
        num_outcomes,
        embedding_size,
        rnn_hidden_units=4,
        fc_hidden_units=10,
        num_treatments=4,
        batchnorm=False,
        rnn_drop_prob=0.0,
        gr_alpha=1.0,
    ):
        super(CRNModel, self).__init__()

        self._input_size = input_size
        self._embedding_size = embedding_size
        self._rnn_hidden_units = rnn_hidden_units
        self._fc_hidden_units = fc_hidden_units
        self._num_treatments = num_treatments
        self._batchnorm = batchnorm
        self._rnn_drop_prob = rnn_drop_prob
        self._gr_alpha = gr_alpha
        self._num_outcomes = num_outcomes

        self._lstm_embedder = LSTMEmbedder(
            input_size=self._input_size,
            output_size=self._embedding_size,
            hidden_units=self._rnn_hidden_units,
            batchnorm=self._batchnorm,
            drop_prob=self._rnn_drop_prob,
        )

        self._treatment_predictor = DNN(
            input_size=self._embedding_size,
            output_size=self._num_treatments,
            num_hidden_layers=0,
            hidden_dim=self._fc_hidden_units,
            flip_gradient=True,
            activation=torch.nn.ELU,
        )

        self._outcome_predictor = DNN(
            input_size=self._embedding_size,
            output_size=self._num_outcomes,
            num_hidden_layers=0,
            hidden_dim=self._fc_hidden_units,
            flip_gradient=True,
            activation=torch.nn.ELU,
        )

    def forward(self, input_sequence):
        lstm_embedding = self._lstm_embedder(input_sequence)
        treatment_logits = self._treatment_predictor(lstm_embedding)
        treatment_pred = torch.nn.functional.softmax(treatment_logits, dim=1)
        outcome_pred = self._outcome_predictor(lstm_embedding)

        return lstm_embedding, treatment_pred, outcome_pred


if __name__ == "__main__":
    crn = CRNModel(
        input_size=10,
        embedding_size=32,
        rnn_hidden_units=16,
        fc_hidden_units=64,
        num_outcomes=10,
    )

    input_data = torch.randn((1, 10, 10))

    out = crn(input_data)

    print("CRN output:")
    for o in out:
        print(o.shape)

    import pdb

    pdb.set_trace()
