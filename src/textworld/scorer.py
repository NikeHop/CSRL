import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import (
    PretrainedEmbeddings,
    GAT,
    CQAttention,
    SelfAttention,
    Attention,
    ParallelGAT,
    ParallelCQAttention,
    ParallelSelfAttention,
    ParallelAttention,
)
from utils.generic import masked_softmax, to_tensor
from utils.layers import ParallelGRU, ParallelLinear


class CommandScorerWithKG(nn.Module):
    def __init__(
        self,
        bidirectional,
        class_tree,
        word_emb,
        hyperbolic_emb,
        numberbatch2hyperbolic,
        graph_emb,
        graph_type,
        n_abstractions,
        hidden_size,
        device,
        opt,
    ):
        super(CommandScorerWithKG, self).__init__()
        self.device = device
        self.hidden_size = hidden_size

        self.class_tree = class_tree
        self.dropout_ratio = 0.0  # *
        self.n_heads = 1  # *
        self.use_hints = True  # *
        self.bidirectional = bidirectional
        self.graph_type = graph_type
        n_factor = 2  # command
        bi_factor = (
            2 if self.bidirectional else 1
        )  # hidden size multiplier when bidirectional is used

        # Additional attributes due to abstraction
        self.n_abstractions = n_abstractions
        self.residual = opt.residual
        self.class_tree = class_tree
        self.hyperbolic = opt.hyperbolic
        self.numberbatch2hyperbolic = numberbatch2hyperbolic

        # Word & Graph Embeddings
        self.word_embedding = PretrainedEmbeddings(word_emb)
        self.word_embedding_size = self.word_embedding.dim

        if self.hyperbolic:
            self.hyperbolic_embeddings = PretrainedEmbeddings(hyperbolic_emb)
            self.word_embedding_size = (
                self.word_embedding.dim + self.hyperbolic_embeddings.dim
            )  
        self.word_embedding_prj = torch.nn.Linear(
            self.word_embedding_size, self.hidden_size, bias=False
        )
        if not self.bidirectional:
            self.word_hint_prj = torch.nn.Linear(
                self.hidden_size * 2, self.hidden_size, bias=False
            )

        self.graph_embedding = None
        if graph_emb is not None and (
            "local" in self.graph_type or "world" in self.graph_type
        ):
            self.graph_embedding = PretrainedEmbeddings(graph_emb, True)
            self.graph_embedding_size = self.graph_embedding.dim
            self.graph_embedding_prj = torch.nn.Linear(
                self.graph_embedding_size, self.hidden_size, bias=False
            )
            if not self.bidirectional:
                self.graph_hint_prj = torch.nn.Linear(
                    self.hidden_size * 2, self.hidden_size, bias=False
                )

        # Encoder for the observation
        self.encoder_gru = ParallelGRU(
            hidden_size,
            hidden_size,
            n_abstractions,
            bidirectional=self.bidirectional,
        )

        # Encoder for the commands
        self.cmd_encoder_gru = ParallelGRU(
            hidden_size,
            hidden_size,
            n_abstractions,
            bidirectional=self.bidirectional,
        )

        # RNN that keeps track of the encoded state over time
        self.state_gru = ParallelGRU(
            hidden_size * bi_factor,
            hidden_size * bi_factor,
            n_abstractions,
            bidirectional=False,
        )

        self.kg_word_encoder_gru = ParallelGRU(
            hidden_size, hidden_size, n_abstractions
        )
        self.kg_graph_encoder_gru = ParallelGRU(
            hidden_size, hidden_size, n_abstractions
        )

        if "local" in self.graph_type or "world" in graph_type:
            self.attention = ParallelCQAttention(
                n_abstractions,
                block_hidden_dim=hidden_size * bi_factor,
                dropout=self.dropout_ratio,
            )
            self.attention_prj = torch.nn.Linear(
                hidden_size * bi_factor * 4, hidden_size * bi_factor, bias=False
            )

        if "world" in self.graph_type:
            n_factor += 1
            self.worldkg_gat = ParallelGAT(
                n_abstractions,
                hidden_size,
                hidden_size,
                self.dropout_ratio,
                alpha=0.2,
                nheads=self.n_heads,
            )
            self.worldkg_attention_prj = ParallelLinear(
                n_abstractions,
                hidden_size * bi_factor * 4,
                hidden_size * bi_factor,
                bias=False,
            )
            self.world_self_attention = ParallelSelfAttention(
                n_abstractions,
                hidden_size * bi_factor,
                hidden_size * bi_factor,
                self.n_heads,
                self.dropout_ratio,
            )
        if "local" in graph_type:
            n_factor += 1
            self.localkg_gat = GAT(
                hidden_size,
                hidden_size,
                self.dropout_ratio,
                alpha=0.2,
                nheads=self.n_heads,
            )
            self.localkg_attention_prj = torch.nn.Linear(
                hidden_size * bi_factor * 4, hidden_size * bi_factor, bias=False
            )
            self.local_self_attention = SelfAttention(
                hidden_size * bi_factor,
                hidden_size * bi_factor,
                self.n_heads,
                self.dropout_ratio,
            )

        self.state_hidden = []
        self.general_attention = ParallelAttention(
            n_abstractions, hidden_size * bi_factor * 2, hidden_size * bi_factor
        )  # General attention from [cmd + obs ==> graph_nodes]
        self.world_attention = None
        self.local_attention = None
        self.obs2kg_attention = torch.nn.Linear(
            hidden_size * bi_factor, hidden_size * bi_factor, bias=False
        )
        self.critic = nn.Linear(hidden_size * bi_factor, 1)

        self.att_cmd = nn.Sequential(
            ParallelLinear(
                n_abstractions,
                hidden_size * bi_factor * n_factor,
                hidden_size * bi_factor,
            ),
            nn.ReLU(),
            ParallelLinear(n_abstractions, hidden_size * bi_factor, 1),
        )
        self.count = 1

    def forward(
        self,
        obs,
        commands,
        mask,
        local_graph,
        local_hints,
        local_adj,
        world_graph,
        world_hints,
        world_adj,
        **kwargs
    ):
        ## Add the classs graph to the forward arguments

        obs = obs.to(self.device)
        commands = commands.to(self.device)
        input_length = obs.size(1)
        batch_size = obs.size(0)
        nb_cmds = commands.size(1)
        cmd_selector_input = []

        # Observed State

        # print(f'Observation: {obs}')
        embedded = self.word_embedding(obs).permute(
            2, 0, 1, 3
        )  # batch x word x emb_size -> batch x word x abstraction x emb_size

        if self.hyperbolic:
            # Transform numberbatch IDs to hyperbolic IDs
            hyperbolic_obs = torch.flatten(
                torch.zeros(obs.shape, dtype=torch.long, device=self.device)
            )
            for i, elem in enumerate(torch.flatten(obs)):
                hyperbolic_obs[i] = self.numberbatch2hyperbolic[elem.item()]

            hyperbolic_obs = hyperbolic_obs.view(obs.shape)
            hyperbolic_embedded = self.hyperbolic_embeddings(hyperbolic_obs).permute(
                2, 0, 1, 3
            )

            # Replace non-entities with 0-embedding
            state_mask = mask[0].permute(2, 0, 1)

            hyperbolic_embedded = hyperbolic_embedded * state_mask.unsqueeze(-1)

            embedded = torch.cat([embedded, hyperbolic_embedded], dim=-1)

        # print(embedded)
        # print(embedded.shape)
        embedded = self.word_embedding_prj(embedded)  # batch x word x hidden  -> bath

        # Encoder GRU

        if self.bidirectional:
            embedded = torch.stack([embedded, torch.flip(embedded, (2,))], dim=1)
            _, encoder_hidden = self.encoder_gru(embedded)  # AxDxBxE
            encoder_hidden = encoder_hidden.transpose(2, 1).flatten(2, 3)
        else:
            _, encoder_hidden = self.encoder_gru(embedded)  # AxBxE

        # encoder_hidden = encoder_hidden.permute(1, 0, 2).reshape(encoder_hidden.shape[1], 1, -1) if \
        #                encoder_hidden.shape[0] == 2 else encoder_hidden

        # State GRU
        if self.state_hidden is None:
            self.state_hidden = torch.zeros_like(encoder_hidden)

        state_output, state_hidden = self.state_gru(
            encoder_hidden.unsqueeze(2), self.state_hidden
        )  # AxBx1xE
        self.state_hidden = state_hidden.detach()

        # Compute state value

        value = self.critic(state_output[0, 0])

        # Commands/Actions
        # Commands Shape: B,number_of_actions,S,A
        # print(f'Commands {commands}')

        cmds_embedding = self.word_embedding(commands).permute(
            3, 0, 1, 2, 4
        )  # AxBxNAxSxE

        if self.hyperbolic:
            hyperbolic_commands = torch.flatten(
                torch.zeros(commands.shape, dtype=torch.long, device=self.device)
            )
            for i, elem in enumerate(torch.flatten(commands)):
                hyperbolic_commands[i] = self.numberbatch2hyperbolic[elem.item()]

            hyperbolic_commands = hyperbolic_commands.view(commands.shape)

            cmds_hyperbolic_embedded = self.hyperbolic_embeddings(
                hyperbolic_commands
            ).permute(3, 0, 1, 2, 4)
            action_mask = mask[1].permute(2, 0, 1).unsqueeze(1)

            cmds_hyperbolic_embedded = cmds_hyperbolic_embedded * action_mask.unsqueeze(
                -1
            )
            cmds_embedding = torch.cat(
                [cmds_embedding, cmds_hyperbolic_embedded], dim=-1
            )

        # print(f'cmds_embedding {cmds_embedding}')
        cmds_embedding = self.word_embedding_prj(cmds_embedding)

        cmds_embedding = cmds_embedding.view(
            self.n_abstractions,
            batch_size * nb_cmds,
            commands.size(2),
            self.hidden_size,
        )  # A x [batch-ncmds] x nentities x hidden_size

        if self.bidirectional:
            cmds_embedding = torch.stack(
                [cmds_embedding, torch.flip(cmds_embedding, (2,))], dim=1
            )
            _, cmds_encoding = self.cmd_encoder_gru(
                cmds_embedding
            )  # A x [batch-ncmds] x hidden_size
            cmds_encoding = cmds_encoding.permute(0, 2, 1, 3).flatten(2, 3)
        else:
            _, cmds_encoding = self.cmd_encoder_gru(cmds_embedding)

        cmds_encoding = cmds_encoding.reshape(
            self.n_abstractions,
            batch_size,
            nb_cmds,
            self.hidden_size * (2 if self.bidirectional else 1),
        )
        cmd_selector_input.append(cmds_encoding)

        # cmds_encoding = cmds_encoding.permute(1, 0, 2).reshape(1, cmds_encoding.shape[1], -1) if \
        #    cmds_encoding.shape[0] == 2 else cmds_encoding
        # cmds_encoding = cmds_encoding.squeeze(0)
        # cmds_encoding = cmds_encoding.view(batch_size, nb_cmds, self.hidden_size * (2 if self.bidirectional else 1))
        # cmd_selector_input.append(cmds_encoding)  # batch x cmds x hidden

        query_encoding = torch.cat(
            [
                cmds_encoding.transpose(1, 0),
                torch.stack([state_hidden.transpose(1, 0)] * nb_cmds, dim=2),
            ],
            dim=-1,
        )  # batch x cmds x hidden*2

        if torch.any(torch.isnan(encoder_hidden)):
            print("error")

        # Local Graph
        localkg_encoding = torch.FloatTensor()
        worldkg_encoding = torch.FloatTensor()
        if "local" in self.graph_type and local_graph.nelement() > 0:
            # graph # num_nodes x entities
            localkg_embedded = self.word_embedding(
                local_graph
            )  # nodes x entities x hidden+
            localkg_embedded = self.word_embedding_prj(
                localkg_embedded
            )  #  nodes x  entities x hidden
            localkg_embedded = localkg_embedded.mean(1)  # nodes x hidden
            localkg_embedded = torch.stack(
                [localkg_embedded] * batch_size, 0
            )  # batch x nodes x hidden
            localkg_encoding = self.localkg_gat(localkg_embedded, local_adj.float())

            if self.use_hints:
                # Get hint with word_embedding ids tensor
                hints_embedded = self.word_embedding(local_hints)
                hints_embedded = self.word_embedding_prj(hints_embedded)
                _, hint_encoding = self.kg_word_encoder_gru(hints_embedded)
                hint_encoding = hint_encoding.squeeze(0)

                localkg_encoding = torch.cat(
                    [
                        localkg_encoding,
                        torch.stack(
                            [hint_encoding.squeeze(1)] * local_graph.shape[0], dim=1
                        ),
                    ],
                    dim=-1,
                )
                if not self.bidirectional:
                    localkg_encoding = self.word_hint_prj(localkg_encoding)

        # World Graph
        if (
            "world" in self.graph_type
            and self.graph_embedding
            and world_graph.nelement() > 0
        ):

            world_graph = world_graph.permute(2, 0, 1)  # n_abstr x nodes x entities
            # graph # num_nodes x entities
            worldkg_embedded = self.graph_embedding(
                world_graph
            )  # n_abstractions x nodes x entities x hidden+
            worldkg_embedded = self.graph_embedding_prj(
                worldkg_embedded
            )  #  nodes x  entities x hidden
            worldkg_embedded = worldkg_embedded.mean(
                2
            )  # n_abstractions x nodes x hidden
            worldkg_embedded = torch.stack(
                [worldkg_embedded] * batch_size, 0
            )  # batch x n_abstractions x nodes x hidden

            worldkg_encoding = self.worldkg_gat(worldkg_embedded, world_adj.float())

            # print('Do we use hints? {self.use_hints}')
            if self.use_hints:
                # Get hint with graph_embedding ids tensor
                world_hints = world_hints.permute(2, 0, 1)
                hints_embedded = self.graph_embedding(world_hints)
                hints_embedded = self.graph_embedding_prj(hints_embedded)

                _, hint_encoding = self.kg_graph_encoder_gru(hints_embedded)

                hint_encoding = hint_encoding.permute(1, 0, 2).unsqueeze(2)

                worldkg_encoding = torch.cat(
                    [
                        worldkg_encoding,
                        torch.cat([hint_encoding] * world_graph.shape[1], dim=2),
                    ],
                    dim=-1,
                )
                if not self.bidirectional:
                    worldkg_encoding = self.graph_hint_prj(worldkg_encoding)

        if (
            "local" in self.graph_type and localkg_encoding.nelement() > 0
        ):  # graphtype = local
            mask = torch.ones(
                (batch_size, 1), device=self.device, requires_grad=False
            ).byte()
            state_hidden = state_hidden.unsqueeze(1)  # batch x 1 x hidden
            obs_encoding = self.attention(
                state_hidden, localkg_encoding, mask, local_adj.sum(dim=2) > 0
            )
            obs_encoding = self.attention_prj(obs_encoding)
            localkg_encoding = self.attention(
                localkg_encoding, state_hidden, local_adj.sum(dim=2) > 0, mask
            )
            localkg_encoding = self.localkg_attention_prj(localkg_encoding)
            state_hidden = obs_encoding.squeeze(1)  # batch x hidden

            local_nodes = local_adj.sum(dim=2)
            m1 = local_nodes.unsqueeze(-1)
            m2 = local_nodes.unsqueeze(1)
            mask_squared = torch.bmm(m1, m2).byte()
            local2obs_encoding, _ = self.local_self_attention(
                localkg_encoding, mask_squared, localkg_encoding, localkg_encoding
            )

            localkg_representation, local_attention = self.general_attention(
                query_encoding, local2obs_encoding
            )
            self.local_attention = local_attention.clone().detach()
            localkg_representation = localkg_representation.squeeze(1)

            cmd_selector_input.append(localkg_representation)

        elif (
            "world" in self.graph_type and worldkg_encoding.nelement() > 0
        ):  # graphtype = world

            mask = torch.ones(
                (batch_size, self.n_abstractions, 1),
                device=self.device,
                requires_grad=False,
            ).byte()
            state_hidden = state_hidden.transpose(1, 0).unsqueeze(2)
            obs_encoding = self.attention(
                state_hidden, worldkg_encoding, mask, world_adj.sum(dim=3) > 0
            )
            obs_encoding = self.attention_prj(obs_encoding)
            worldkg_encoding = self.attention(
                worldkg_encoding, state_hidden, world_adj.sum(dim=3) > 0, mask
            )
            worldkg_encoding = self.worldkg_attention_prj(
                worldkg_encoding.transpose(1, 0)
            ).transpose(1, 0)

            state_hidden = obs_encoding.transpose(1, 0).squeeze(2)

            world_nodes = world_adj.sum(dim=3)  # batch x nworld
            m1 = world_nodes.unsqueeze(-1)
            m2 = world_nodes.unsqueeze(2)

            mask_squared = torch.matmul(m1, m2).byte()
            world2obs_encoding, _ = self.world_self_attention(
                worldkg_encoding, mask_squared, worldkg_encoding, worldkg_encoding
            )
            worldkg_representation, world_attention = self.general_attention(
                query_encoding, world2obs_encoding
            )

            self.world_attention = world_attention.clone().detach()

            cmd_selector_input.append(worldkg_representation.transpose(1, 0))

        self.count += 1

        # Concatenate the observed state (required) and command (required) and scored command history (optional) encodings
        # with kg-based encodings for commnads (optional) and scored command history (optional).
        # State rpresentaton for all types of agents

        cmd_selector_input.append(
            torch.stack([state_hidden] * nb_cmds, 2)
        )  # A x batch x cmds x hidden
        cmd_selector_new_input = torch.cat(
            cmd_selector_input, dim=-1
        )  # batch x ncmds x [hidden*nfactor]

        # Compute one score per command.
        scores = self.att_cmd(cmd_selector_new_input).squeeze(
            -1
        )  # n_abstraction x batch x ncmds

        # Here you need to adapat as well
        if self.residual == "residual2":
            scores = F.softplus(scores)
            base_prob = F.softmax(scores[-1], dim=-1)
            for score in reversed(scores[:-1]):
                base_prob = score * base_prob / torch.sum(score * base_prob, dim=-1)
            probs = base_prob
            index = probs.multinomial(num_samples=1).unsqueeze(0)

        else:
            summed_scores = torch.sum(scores, dim=0)

            # Sampling action with abstractions
            probs = masked_softmax(
                summed_scores, commands[:, :, :, 0].sum(dim=2) > 0, dim=1
            )  # batch x cmds

            index = torch.tensor(
                random.choices(
                    list(range(probs.shape[1])), weights=list(probs.flatten())
                )
            ).reshape(1, 1)

            # with open("/var/scratch/nrhopner/experiments/priorKG/TWC/commonsense-rl/debug_info_4.txt","a+") as f:
            #            f.write(f"Prob {probs}\n")
            #            f.write(f"Index: {index}\n")

            # print(index)

            # index = probs.multinomial(num_samples=1).unsqueeze(0)

            # print(index)

        return scores, index, value

    def reset_hidden(self, batch_size):
        self.state_hidden = torch.zeros(
            self.n_abstractions,
            batch_size,
            self.hidden_size * (2 if self.bidirectional else 1),
            device=self.device,
        )

    def reset_hidden_per_batch(self, batch_id):
        self.state_hidden[:, batch_id, :] = torch.zeros(
            self.n_abstractions,
            self.hidden_size * (2 if self.bidirectional else 1),
            device=self.device,
        )
