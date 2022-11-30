# Layers that are necessary for parallelization with abstraction
import torch
import torch.nn as nn

class ParallelGRU(nn.Module):
    def __init__(
        self,
        input_size:int,
        hidden_size:int,
        n_abstractions:int,
        bidirectional:bool=False
    ):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        if self.bidirectional:
            input_shape_weight = (n_abstractions, 2, input_size, hidden_size)
            input_shape_bias = (n_abstractions, 2, hidden_size)
            hidden_shape_weight = (n_abstractions, 2, hidden_size, hidden_size)
            hidden_shape_bias = (n_abstractions, 2, hidden_size)
        else:
            input_shape_weight = (n_abstractions, input_size, hidden_size)
            input_shape_bias = (n_abstractions, hidden_size)
            hidden_shape_weight = (n_abstractions, hidden_size, hidden_size)
            hidden_shape_bias = (n_abstractions, hidden_size)

        self.input2r = nn.Parameter(torch.zeros(*input_shape_weight))
        self.input2z = nn.Parameter(torch.zeros(*input_shape_weight))
        self.input2n = nn.Parameter(torch.zeros(*input_shape_weight))
        self.input2r_bias = nn.Parameter(torch.zeros(*input_shape_bias))
        self.input2z_bias = nn.Parameter(torch.zeros(*input_shape_bias))
        self.input2n_bias = nn.Parameter(torch.zeros(*input_shape_bias))

        self.hidden2r = nn.Parameter(torch.zeros(*hidden_shape_weight))
        self.hidden2z = nn.Parameter(torch.zeros(*hidden_shape_weight))
        self.hidden2n = nn.Parameter(torch.zeros(*hidden_shape_weight))
        self.hidden2r_bias = nn.Parameter(torch.zeros(*hidden_shape_bias))
        self.hidden2z_bias = nn.Parameter(torch.zeros(*hidden_shape_bias))
        self.hidden2n_bias = nn.Parameter(torch.zeros(*hidden_shape_bias))

        # Initialize weights
        nn.init.xavier_uniform_(self.input2r)
        nn.init.xavier_uniform_(self.input2z)
        nn.init.xavier_uniform_(self.input2n)
        nn.init.xavier_uniform_(self.hidden2r)
        nn.init.xavier_uniform_(self.hidden2z)
        nn.init.xavier_uniform_(self.hidden2n)

    def forward(self, x, h=None):
        """
        Not bidirectional:
        x (tensor): AxBxSxE
        h (tensor): AxBxE

        Bidirectional:
        x (tensor): AxBxDxSxE
        h (tensor): AxBxDxE
        """

        if not self.bidirectional:
            A, B, S, E = x.shape
            if not isinstance(h, torch.Tensor):
                h = torch.zeros(A, B, E).to(x.device)

            outputs = []
            for i in range(S):
                step_input = x[:, :, i]  # AxBxE
                r = torch.sigmoid(
                    torch.bmm(step_input, self.input2r)
                    + self.input2r_bias.unsqueeze(1)
                    + torch.bmm(h, self.hidden2r)
                    + self.hidden2r_bias.unsqueeze(1)
                )
                z = torch.sigmoid(
                    torch.bmm(step_input, self.input2z)
                    + self.input2z_bias.unsqueeze(1)
                    + torch.bmm(h, self.hidden2z)
                    + self.hidden2z_bias.unsqueeze(1)
                )
                n = torch.tanh(
                    torch.bmm(step_input, self.input2n)
                    + self.input2n_bias.unsqueeze(1)
                    + r
                    * (torch.bmm(h, self.hidden2n) + self.hidden2n_bias.unsqueeze(1))
                )
                h = (1 - z) * n + z * h
                outputs.append(h)

            outputs = torch.stack(outputs, dim=0).permute(1, 2, 0, 3)
        else:
            A, D, B, S, E = x.shape
            if not isinstance(h, torch.Tensor):
                h = torch.zeros(A, D, B, E).to(x.device)

            outputs = []
            for i in range(S):
                step_input = x[:, :, :, i]  # AxDxBxE
                r = torch.sigmoid(
                    torch.matmul(step_input, self.input2r)
                    + self.input2r_bias.unsqueeze(2)
                    + torch.matmul(h, self.hidden2r)
                    + self.hidden2r_bias.unsqueeze(2)
                )
                z = torch.sigmoid(
                    torch.matmul(step_input, self.input2z)
                    + self.input2z_bias.unsqueeze(2)
                    + torch.matmul(h, self.hidden2z)
                    + self.hidden2z_bias.unsqueeze(2)
                )
                n = torch.tanh(
                    torch.matmul(step_input, self.input2n)
                    + self.input2n_bias.unsqueeze(2)
                    + r
                    * (torch.matmul(h, self.hidden2n) + self.hidden2n_bias.unsqueeze(2))
                )
                h = (1 - z) * n + z * h
                outputs.append(h)

            outputs = torch.stack(outputs, dim=0).permute(1, 3, 2, 0, 4)

        return outputs, h


class ParallelLinear(nn.Module):
    def __init__(self, n_abstractions:int, input:int, output:int, bias=True):

        super().__init__()

        self.output_dim = output
        self.weight_matrix = nn.Parameter(torch.zeros((n_abstractions, input, output)))
        self.bias = nn.Parameter(torch.zeros(n_abstractions, output))

        nn.init.xavier_uniform(self.weight_matrix)

        self.use_bias = bias

    def forward(self, x):
        """
        x (tensor): AxBxNAxE
        """

        A, B, NA, E = x.shape
        x = x.reshape(A, B * NA, E)
        output = torch.bmm(x, self.weight_matrix)
        if self.use_bias:
            output = output + self.bias.unsqueeze(1)
        output = output.reshape(A, B, NA, self.output_dim)
        return output
