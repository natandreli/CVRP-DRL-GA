import torch


class PointerNetwork(torch.nn.Module):
    """
    Pointer Network for sequence-to-sequence problems.

    Uses attention mechanism to "point" to input elements.
    Adapted for CVRP: encodes customer features and points to next customer.
    """

    def __init__(
        self,
        input_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize Pointer Network.

        Args:
            input_dim: Dimension of input features per customer
            embedding_dim: Dimension for embedding layer
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        # Encoder: embed customer features
        self.embedding = torch.nn.Linear(input_dim, embedding_dim)

        # LSTM encoder
        self.encoder_lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Decoder LSTM
        self.decoder_lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention mechanism (Pointer)
        self.attention_linear = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.pointer_v = torch.nn.Linear(hidden_dim, 1, bias=False)

        self.dropout = torch.nn.Dropout(dropout)

    def forward(
        self,
        customer_features: torch.Tensor,
        decoder_input: torch.Tensor,
        hidden: tuple,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, tuple]:
        """
        Forward pass.

        Args:
            customer_features: [batch, num_customers, input_dim]
            decoder_input: [batch, 1, embedding_dim]
            hidden: (h, c) tuple from previous step
            mask: [batch, num_customers] - True for invalid actions

        Returns:
            - Pointer logits: [batch, num_customers]
            - Hidden state: (h, c) tuple
        """
        batch_size, num_customers, _ = customer_features.shape

        # Encode customers
        embedded = self.dropout(
            torch.nn.functional.relu(self.embedding(customer_features))
        )

        # Decode one step
        decoder_output, hidden_new = self.decoder_lstm(decoder_input, hidden)
        decoder_output = decoder_output.squeeze(1)  # [batch, hidden_dim]

        # Attention: compute scores for each customer
        # Expand decoder output to match customer dimension
        decoder_expanded = decoder_output.unsqueeze(1).expand(
            batch_size, num_customers, self.hidden_dim
        )

        # Concatenate decoder output with encoded customers
        attention_input = torch.cat([decoder_expanded, embedded], dim=-1)
        attention_hidden = torch.tanh(self.attention_linear(attention_input))

        # Pointer logits
        logits = self.pointer_v(attention_hidden).squeeze(-1)  # [batch, num_customers]

        # Apply mask (set invalid actions to -inf)
        if mask is not None:
            logits = logits.masked_fill(mask, float("-inf"))

        return logits, hidden_new

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """Initialize hidden state for LSTM."""
        h = torch.zeros(
            self.encoder_lstm.num_layers, batch_size, self.hidden_dim, device=device
        )
        c = torch.zeros(
            self.encoder_lstm.num_layers, batch_size, self.hidden_dim, device=device
        )
        return (h, c)


class ActorCriticNetwork(torch.nn.Module):
    """
    Actor-Critic architecture for CVRP with Pointer Networks.

    - Actor: Pointer Network que genera política (qué cliente visitar)
    - Critic: Red que estima el valor del estado actual
    """

    def __init__(
        self,
        customer_feature_dim: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.1,
    ):
        """
        Initialize Actor-Critic.

        Args:
            customer_feature_dim: Features per customer (visited, demand, distance, x, y)
            embedding_dim: Embedding dimension
            hidden_dim: Hidden dimension for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()

        self.customer_feature_dim = customer_feature_dim

        # Actor: Pointer Network
        self.actor = PointerNetwork(
            input_dim=customer_feature_dim,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        # Critic: Value function estimator
        # Takes state representation and outputs scalar value
        self.critic_embedding = torch.nn.Linear(customer_feature_dim, embedding_dim)
        self.critic_lstm = torch.nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.critic_value = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim // 2, 1),
        )

        self.dropout = torch.nn.Dropout(dropout)

    def forward_actor(
        self,
        customer_features: torch.Tensor,
        decoder_input: torch.Tensor,
        hidden: tuple,
        mask: torch.Tensor = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple]:
        """
        Actor forward pass.

        Args:
            customer_features: [batch, num_customers, feature_dim]
            decoder_input: [batch, 1, embedding_dim]
            hidden: LSTM hidden state
            mask: Invalid action mask

        Returns:
            - Action probabilities: [batch, num_customers]
            - Log probabilities: [batch, num_customers]
            - New hidden state
        """
        logits, hidden_new = self.actor(customer_features, decoder_input, hidden, mask)

        # Softmax to get probabilities
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        return probs, log_probs, hidden_new

    def forward_critic(self, customer_features: torch.Tensor) -> torch.Tensor:
        """
        Critic forward pass: estimate state value.

        Args:
            customer_features: [batch, num_customers, feature_dim]

        Returns:
            State value: [batch, 1]
        """
        # Embed and encode
        embedded = self.dropout(
            torch.nn.functional.relu(self.critic_embedding(customer_features))
        )

        # LSTM encoding
        lstm_out, _ = self.critic_lstm(embedded)

        # Use last output for value estimation
        last_output = lstm_out[:, -1, :]  # [batch, hidden_dim]

        # Value prediction
        value = self.critic_value(last_output)

        return value

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple:
        """Initialize hidden state for actor."""
        return self.actor.init_hidden(batch_size, device)
