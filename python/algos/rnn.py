

class RNNUnit(nn.Module):
    def __init__(self, emb_dim):
        super.__init__()

        self.Uh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.Wh = nn.Parameter(torch.randn(emb_dim, emb_dim))
        self.b = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x, h):
        return torch.tanh(x @ self.Wh + h @ self.Uh + self.b)


class RNN(nn.Module):
    def __init__(self, emb_dim, num_layers):
        super.__init__()

        self.emb_dim = emb_dim
        self.num_layers = num_layers
        self.rnn_units = nn.ModuleList([RNNUnit(emb_dim) for _ in range(num_layers)])

    def forward(self, x):
        batch_size, seq_len, emb_dim = x.shape

        h_prev = [
            torch.zeros(batch_size, emb_dim, device=x.device)
            for _ in range(self.num_layers)
        ]

        outputs = []

        # TODO too slow!
        for t in range(seq_len):
            input_t = x[:,t]

            for l, rnn_unit in enumerate(self.rnn_units):
                h_new = rnn_unit(input_t, h_prev[l])
                h_prev[l] = h_new

                input_t = h_new

            outputs.append(input_t)

        return torch.stack(outputs, dim=1)
                
            
