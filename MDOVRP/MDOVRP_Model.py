import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv

class MDOVRPModel(nn.Module):

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params

        self.encoder = Encoder(**model_params)
        self.decoder = Decoder(**model_params)
        self.encoded_nodes = None
        # shape: (batch_size, node_size, embedding_dim)
        self.depot_size = None
        self.encoded_depots = None
        self.encoded_customers = None

    def pre_forward(self, reset_state):
        depot_x_y = reset_state.depot_x_y
        # shape: (batch_size, depot_size, 2)
        customer_x_y = reset_state.customer_x_y
        # shape: (batch_size, customer_size, 2)
        customer_demand = reset_state.customer_demand
        # shape: (batch_size, customer_size)
        self.depot_size = reset_state.depot_size
        customer_x_y_demand = torch.cat((customer_x_y, customer_demand[:, :, None]), dim=2)
        # shape: (batch_size, customer_size, 3)
        self.encoded_nodes, self.encoded_depots, self.encoded_customers = self.encoder(depot_x_y, customer_x_y_demand)
        # shape: (batch_size, node_size, embedding_dim)
        self.decoder.set_k_v(self.encoded_nodes, self.encoded_depots, self.encoded_customers, reset_state)

    def forward(self, state):
        batch_size = state.batch_idx.size(0)
        mt_size = state.mt_idx.size(1)
        if state.selected_count == 0:  # first move depot
            selected = torch.zeros(size=(batch_size, mt_size), dtype=torch.long)
            probability = torch.ones(size=(batch_size, mt_size))
        elif state.selected_count == 1:  # second move mt
            selected = torch.arange(1, mt_size + 1)[None, :].expand(batch_size, mt_size)
            probability = torch.ones(size=(batch_size, mt_size))
        else:
            prob = self.decoder(self.encoded_nodes, state, mask=state.mask)
            # shape: (batch_size, mt_size, node_size)
            if self.training or self.model_params['sample']:
                while True:
                    with torch.no_grad():
                        selected = prob.reshape(batch_size * mt_size, -1).multinomial(1).squeeze(dim=1) \
                            .reshape(batch_size, mt_size)
                        # shape: (batch_size, mt_size)
                    probability = prob[state.batch_idx, state.mt_idx, selected].reshape(batch_size, mt_size)
                    # shape: (batch_size, mt_size)
                    if (probability != 0).all():
                        break
            else:
                selected = prob.argmax(dim=2)
                # shape: (batch_size, mt_size)
                probability = None
        return selected, probability

class Encoder(nn.Module):
    """Hyper-Relation Fusion Encoder for MDVRP
    Integrates heterogeneous graph attention (HeGNN) and homogeneous graph convolution (HoGNN)
    with transformer-based feature enhancement

    Args:
        model_params (dict): Configuration parameters including:
            - embedding_dim: Dimension of node embeddings
            - encoder_layer_num: Number of transformer layers
            - gat_heads: Number of GAT attention heads
            - dropout: Dropout probability
            - hognn_layers: Number of HoGNN layers
            - hegnn_layers: Number of HeGNN layers
    """

    def __init__(self, ** model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = model_params['embedding_dim']
        self.encoder_layer_num = model_params['encoder_layer_num']
        self.edge_type_num = 2  # 0: Depot-Customer, 1: Customer-Customer

        # Feature embedding layers
        self.embedding_depot = nn.Linear(2, self.embedding_dim)  # (x,y) -> embedding
        self.embedding_customer = nn.Linear(3, self.embedding_dim)  # (x,y,demand) -> embedding

        # Heterogeneous Graph Attention Network
        self.hegnn_layers = nn.ModuleList([
            HeteroGATLayer(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim,
                edge_types=self.edge_type_num,
                heads=model_params.get('gat_heads', 4),
                dropout=model_params.get('dropout', 0.1)
            ) for _ in range(model_params.get('hegnn_layers', 2))
        ])

        # Homogeneous Graph Convolution Network
        self.hognn_layers = nn.ModuleList([
            HomoGCNLayer(
                in_channels=self.embedding_dim,
                out_channels=self.embedding_dim
            ) for _ in range(model_params.get('hognn_layers', 2))
        ])

        # Multi-modal feature fusion
        self.fusion_proj = nn.Linear(3 * self.embedding_dim, self.embedding_dim)

        # Transformer feature enhancer
        self.transformer_layers = nn.ModuleList([
            EncoderLayer(**model_params)
            for _ in range(self.encoder_layer_num)
        ])

    def forward(self, depot_x_y, customer_x_y_demand):
        """
        Processing pipeline:
        1. Embed raw features
        2. Construct heterogeneous graph
        3. Apply HeGNN and HoGNN
        4. Fuse multi-relation features
        5. Enhance with transformer

        Args:
            depot_x_y: (batch_size, depot_size, 2) Depot coordinates
            customer_x_y_demand: (batch_size, customer_size, 3) Customer features

        Returns:
            fused_emb: (batch_size, node_size, embedding_dim) Integrated node embeddings
            embedded_depot: (batch_size, depot_size, embedding_dim) Depot-specific embeddings
            embedded_customers: (batch_size, customer_size, embedding_dim) Customer-specific embeddings
        """
        # Feature embedding
        embedded_depots = self.embedding_depot(depot_x_y)  # (batch, depot_size, emb_dim)
        embedded_customers = self.embedding_customer(customer_x_y_demand)  # (batch, customer_size, emb_dim)
        batch_size, depot_size = embedded_depots.shape[:2]
        customer_size = embedded_customers.shape[1]

        # Construct full node embeddings
        node_emb = torch.cat([embedded_depots, embedded_customers], dim=1)  # (batch, node_size, emb_dim)
        orig_emb = node_emb.clone()

        # Generate heterogeneous graph structure
        edge_index, edge_type = self.build_hetero_edges(depot_x_y, depot_size, customer_size, batch_size)

        # Heterogeneous graph processing
        he_emb = node_emb
        for he_layer in self.hegnn_layers:
            he_emb = he_layer(he_emb, edge_index, edge_type)  # (batch, node_size, emb_dim)

        # Homogeneous graph processing
        ho_emb = orig_emb
        for ho_layer in self.hognn_layers:
            ho_emb = ho_layer(ho_emb)  # (batch, node_size, emb_dim)

        # Multi-relation feature fusion
        fused_emb = self.fusion_proj(
            torch.cat([orig_emb, he_emb, ho_emb], dim=-1)  # (batch, node_size, 3*emb_dim)
        )  # (batch, node_size, emb_dim)

        # Transformer feature enhancement
        for layer in self.transformer_layers:
            fused_emb = layer(fused_emb)  # (batch, node_size, emb_dim)

        # Decompose features
        embedded_depot = fused_emb[:, :depot_size, :]  # (batch, depot_size, emb_dim)
        embedded_customers = fused_emb[:, depot_size:, :]  # (batch, customer_size, emb_dim)

        return fused_emb, embedded_depot, embedded_customers

    def build_hetero_edges(self, depot_x_y, depot_size, customer_size, batch_size):
        """Constructs heterogeneous graph connections

        Returns:
            edge_index: (batch_size, 2, total_edges) Edge connections
            edge_type: (batch_size, total_edges) Edge types (0/1)
        """
        node_size = depot_size + customer_size
        edge_list = []
        edge_types = []

        # Depot-Customer connections (type 0)
        for d in range(depot_size):
            for c in range(customer_size):
                edge_list.append((d, depot_size + c))
                edge_types.append(0)

        # Customer-Customer connections (type 1)
        for c1 in range(customer_size):
            for c2 in range(customer_size):
                if c1 != c2:
                    edge_list.append((depot_size + c1, depot_size + c2))
                    edge_types.append(1)

        # Convert to tensor format
        edge_index = torch.tensor(edge_list).t().contiguous()  # (2, total_edges)
        edge_type = torch.tensor(edge_types, dtype=torch.long)  # (total_edges,)

        # Batch dimension expansion
        edge_index = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)  # (batch, 2, total_edges)
        edge_type = edge_type.unsqueeze(0).repeat(batch_size, 1)  # (batch, total_edges)

        return edge_index.to(depot_x_y.device), edge_type.to(depot_x_y.device)


class HeteroGATLayer(nn.Module):
    """Enhanced Heterogeneous Graph Attention Layer with:
    - Batch-aware edge indexing
    - Dynamic padding for variable edge numbers
    - Edge-type specific parameters
    - Device consistency checks
    - Tensor shape validation

    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
        edge_types (int): Number of edge types
        heads (int): Number of attention heads
        dropout (float): Dropout probability
    """

    def __init__(self, in_channels, out_channels, edge_types, heads=4, dropout=0.1):
        super().__init__()
        self.edge_types = edge_types
        self.heads = heads
        self.out_channels = out_channels

        # Edge-type specific GAT modules
        self.type_attentions = nn.ModuleList([
            GATConv(
                in_channels,
                out_channels // heads,
                heads=heads,
                dropout=dropout,
                add_self_loops=False
            ) for _ in range(edge_types)
        ])

        # Regularization components
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)

        # Shape validation parameters
        self.last_input_shape = None
        self.last_edge_shape = None

    def forward(self, x, edge_index, edge_type):
        """
        Args:
            x: (batch_size, num_nodes, in_channels) Node features
            edge_index: (batch_size, 2, num_edges) Edge connections
            edge_type: (batch_size, num_edges) Edge type labels

        Returns:
            (batch_size, num_nodes, out_channels) Updated node features
        """
        # Input validation
        self._validate_inputs(x, edge_index, edge_type)

        batch_size, num_nodes, _ = x.shape
        out = torch.zeros_like(x)

        # Device consistency check
        self._check_device_consistency(x, edge_index, edge_type)

        for etype in range(self.edge_types):
            # Generate mask for current edge type
            type_mask = (edge_type == etype)  # (batch_size, num_edges)

            if not type_mask.any():
                continue  # Skip edge types with no connections

            # Extract valid edges with dynamic padding
            valid_edges = self._get_valid_edges(edge_index, type_mask)

            # Process each sample in batch
            batch_output = []
            for b in range(batch_size):
                adj = self._build_sparse_adjacency(valid_edges[b], num_nodes)
                gat_out = self.type_attentions[etype](
                    x[b],  # (num_nodes, in_channels)
                    adj  # (2, num_valid_edges)
                )
                batch_output.append(gat_out)

            # Aggregate batch outputs
            out += torch.stack(batch_output, dim=0)  # (batch_size, num_nodes, out_channels)

        # Final processing
        return self._finalize_output(x, out)

    def _validate_inputs(self, x, edge_index, edge_type):
        """Validate input tensor dimensions"""
        assert x.dim() == 3, f"Expected 3D node features, got {x.dim()}D"
        assert edge_index.dim() == 3, f"Expected 3D edge index, got {edge_index.dim()}D"
        assert edge_type.dim() == 2, f"Expected 2D edge type, got {edge_type.dim()}D"
        self.last_input_shape = (x.shape, edge_index.shape, edge_type.shape)

    def _check_device_consistency(self, *tensors):
        """Ensure all tensors are on same device"""
        devices = {t.device for t in tensors}
        assert len(devices) == 1, f"Tensors on multiple devices: {devices}"

    def _get_valid_edges(self, edge_index, type_mask):
        """Extract valid edges with dynamic padding

        Returns:
            (batch_size, 2, max_edges_per_batch)
        """
        batch_size, num_edges = type_mask.shape
        valid_indices = [torch.nonzero(type_mask[b], as_tuple=True)[0]
                         for b in range(batch_size)]

        # Calculate max edges for padding
        max_edges = max(len(idx) for idx in valid_indices)

        # Create padded index tensor
        padded_indices = torch.zeros((batch_size, max_edges),
                                     dtype=torch.long,
                                     device=edge_index.device)
        for b in range(batch_size):
            valid_len = len(valid_indices[b])
            if valid_len > 0:
                padded_indices[b, :valid_len] = valid_indices[b]

        # Gather valid edges
        return torch.gather(
            edge_index,
            dim=2,
            index=padded_indices.unsqueeze(1).expand(-1, 2, -1)
        )

    def _build_sparse_adjacency(self, edge_index, num_nodes):
        """Convert edge index to sparse adjacency matrix"""
        adj = torch.zeros(num_nodes, num_nodes,
                          device=edge_index.device)
        if edge_index.shape[1] > 0:
            src, dst = edge_index
            adj[src, dst] = 1
        return adj.nonzero(as_tuple=False).t()

    def _finalize_output(self, original_x, new_output):
        """Apply normalization and residual connection"""
        # Shape validation
        assert original_x.shape[:2] == new_output.shape[:2], \
            f"Shape mismatch: {original_x.shape} vs {new_output.shape}"

        # Residual connection with dropout
        return self.norm(original_x + self.dropout(new_output))

    def __repr__(self):
        return (f"HeteroGATLayer(in={self.type_attentions[0].in_channels}, "
                f"out={self.out_channels}, types={self.edge_types}, "
                f"heads={self.heads})")


class HomoGCNLayer(nn.Module):
    """Enhanced Homogeneous Graph Convolution Layer with:
    - Full connectivity support
    - Edge index caching
    - Device consistency checks
    - Input validation
    - Performance optimization

    Args:
        in_channels (int): Input feature dimension
        out_channels (int): Output feature dimension
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Disable automatic self-loop addition to manually handle full connectivity
        self.gcn = GCNConv(in_channels, out_channels, add_self_loops=False)
        self.norm = nn.LayerNorm(out_channels)

        # Cache for fully connected edge indices
        self.cached_edge_index = None
        self.last_num_nodes = None  # Track last processed node count for cache validation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input through GCN layer with full connectivity

        Args:
            x: Tensor of shape (batch_size, num_nodes, in_channels) containing node features

        Returns:
            Tensor of shape (batch_size, num_nodes, out_channels) with updated features
        """
        # Input dimension validation
        self._validate_input(x)

        batch_size, num_nodes, _ = x.shape
        device = x.device

        # Generate/Cache fully connected edge indices
        edge_index = self._get_full_connectivity(num_nodes, device)

        # Process each sample in batch
        batch_outputs = []
        for b in range(batch_size):
            # Apply GCN with precomputed edge indices
            batch_outputs.append(self.gcn(x[b], edge_index))

        # Combine results and apply residual connection
        return self._apply_residual(x, torch.stack(batch_outputs, dim=0))

    def _validate_input(self, x: torch.Tensor) -> None:
        """Validate input tensor dimensions

        Args:
            x: Input tensor to validate
        Raises:
            ValueError: If input is not 3-dimensional
        """
        if x.dim() != 3:
            raise ValueError(f"Expected 3D tensor (batch, nodes, features), got {x.dim()}D")

    def _get_full_connectivity(self, num_nodes: int, device: torch.device) -> torch.Tensor:
        """Generate fully connected edge indices with caching mechanism

        Args:
            num_nodes: Number of nodes in graph
            device: Target device for tensor creation
        Returns:
            Edge index tensor of shape (2, num_nodes²)
        """
        if self.last_num_nodes != num_nodes or self.cached_edge_index is None:
            # Generate all possible node pairs using meshgrid
            rows, cols = torch.meshgrid(
                torch.arange(num_nodes, device=device),
                torch.arange(num_nodes, device=device),
                indexing='ij'
            )

            # Convert to edge index format (2, num_nodes²)
            self.cached_edge_index = torch.stack([
                rows.flatten(),  # Source nodes
                cols.flatten()  # Target nodes
            ], dim=0)

            self.last_num_nodes = num_nodes  # Update cache tracking

        return self.cached_edge_index

    def _apply_residual(self, original: torch.Tensor, updated: torch.Tensor) -> torch.Tensor:
        """Apply residual connection and layer normalization

        Args:
            original: Original input tensor
            updated: Updated tensor from GCN processing
        Returns:
            Normalized output tensor
        Raises:
            RuntimeError: If tensor shapes mismatch
        """
        # Shape consistency check
        if original.shape != updated.shape:
            raise RuntimeError(
                f"Shape mismatch: original {original.shape} vs updated {updated.shape}"
            )

        return self.norm(original + updated)

    def extra_repr(self) -> str:
        """Generate formal string representation for module

        Returns:
            String containing layer configuration details
        """
        return f"in={self.gcn.in_channels}, out={self.gcn.out_channels}, " \
               f"cached_edges={self.cached_edge_index.shape if self.cached_edge_index is not None else 'None'}"


class EncoderLayer(nn.Module):
    """Transformer Encoder Layer with multi-head self-attention

    Args:
        model_params (dict): Contains:
            - embedding_dim: Feature dimension
            - head_num: Number of attention heads
            - qkv_dim: Dimension per attention head
    """

    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = model_params['embedding_dim']
        self.head_num = model_params['head_num']
        self.qkv_dim = model_params['qkv_dim']

        # Multi-head Attention Projections
        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        # Sub-modules
        self.norm1 = Norm(**model_params)  # Post-Attention Norm (InstanceNorm1d)
        self.ff = FF(**model_params)       # Feed Forward Network
        self.norm2 = Norm(**model_params)  # Post-FFN Norm (InstanceNorm1d)

    def forward(self, out: torch.Tensor) -> torch.Tensor:
        """
        Args:
            out: Input tensor (batch_size, node_size, embedding_dim)
        Returns:
            (batch_size, node_size, embedding_dim)
        """
        # ====== Multi-head Self-Attention ======
        # Projections
        q = multi_head_qkv(self.Wq(out), self.head_num)  # (batch_size, head_num, node_size, qkv_dim)
        k = multi_head_qkv(self.Wk(out), self.head_num)  # (batch_size, head_num, node_size, qkv_dim)
        v = multi_head_qkv(self.Wv(out), self.head_num)  # (batch_size, head_num, node_size, qkv_dim)

        # Attention Computation (non-mask)
        attn_out = multi_head_attention(q, k, v)  # (batch_size, node_size, head_num*qkv_dim)
        attn_out = self.multi_head_combine(attn_out)  # (batch_size, node_size, embedding_dim)

        # Residual + InstanceNorm1d
        out1 = self.norm1(attn_out, out)  # 输入顺序：(new_tensor, residual)

        # ====== Position-wise Feed Forward ======
        ff_out = self.ff(out1)  # (batch_size, node_size, embedding_dim)

        # Residual + InstanceNorm1d
        out2 = self.norm2(ff_out, out1)  # 输入顺序：(new_tensor, residual)

        return out2


class EncoderLayer(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']

        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)

        self.norm1 = Norm(**model_params)
        self.ff = FF(**model_params)
        self.norm2 = Norm(**model_params)

    def forward(self, out):
        # shape: (batch_size, node_size, embedding_dim)
        q = multi_head_qkv(self.Wq(out), head_num=self.head_num)
        k = multi_head_qkv(self.Wk(out), head_num=self.head_num)
        v = multi_head_qkv(self.Wv(out), head_num=self.head_num)
        # shape: (batch_size, head_num, node_size, qkv_dim)
        out_concat = multi_head_attention(q, k, v)
        # shape: (batch_size, node_size, head_num * qkv_dim)
        multi_head_out = self.multi_head_combine(out_concat)
        # shape: (batch_size, node_size, embedding_dim)
        out1 = self.norm1(out, multi_head_out)
        out2 = self.ff(out1)
        out3 = self.norm2(out1, out2)
        return out3
        # shape :(batch_size, node_size, embedding_dim)


class Decoder(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.model_params = model_params
        self.embedding_dim = self.model_params['embedding_dim']
        self.head_num = self.model_params['head_num']
        self.qkv_dim = self.model_params['qkv_dim']
        self.clip = self.model_params['clip']
        self.Wq = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_depot = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_depot = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wk_customer = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.Wv_customer = nn.Linear(self.embedding_dim, self.head_num * self.qkv_dim, bias=False)
        self.multi_head_combine = nn.Linear(self.head_num * self.qkv_dim, self.embedding_dim)
        self.k = None
        self.v = None
        self.depots_key = None
        self.add_key = None
        self.nodes_key = None
        self.q = None
        self.k_depot = None
        self.v_depot = None
        self.k_customer = None
        self.v_customer = None
        self.depot_size = None
        self.customer_size = None
        self.node_size = None

    def set_k_v(self, encoded_nodes, encoded_depots, encoded_customers, reset_state):
        self.depot_size = reset_state.depot_size
        self.customer_size = reset_state.customer_size
        self.node_size = self.depot_size + self.customer_size

        self.k = multi_head_qkv(self.Wk(encoded_nodes), head_num=self.head_num)
        self.v = multi_head_qkv(self.Wv(encoded_nodes), head_num=self.head_num)
        # shape: (batch_size, head_num, node_size, qkv_dim)

        self.nodes_key = encoded_nodes.transpose(1, 2)
        # shape: (batch_size, embedding_dim, node_size)

        # encoded_depots = encoded_nodes[:, :self.depot_size, :].contiguous()
        # shape: (batch_size, depot_size, embedding_dim)
        self.k_depot = multi_head_qkv(self.Wk_depot(encoded_depots), head_num=self.head_num)
        self.v_depot = multi_head_qkv(self.Wv_depot(encoded_depots), head_num=self.head_num)
        # shape: (batch_size, head_num, depot_size, qkv_dim)

        self.k_customer = multi_head_qkv(self.Wk_depot(encoded_customers), head_num=self.head_num)
        self.v_customer = multi_head_qkv(self.Wv_depot(encoded_customers), head_num=self.head_num)
        # shape: (batch_size, head_num, customer_size, qkv_dim)

    def forward(self, encoded_nodes, state, mask):
        # mask shape: (batch_size, mt_size, node_size)
        q = get_encoding(encoded_nodes, state)
        # shape: (batch_size, mt_size, embedding_dim)
        self.q = multi_head_qkv(self.Wq(q), head_num=self.head_num)
        # shape: (batch_size, head_num, mt_size, qkv_dim)
        attention_nodes = multi_head_attention(self.q, self.k, self.v, rank3_mask=mask)
        attention_depots = multi_head_attention(self.q, self.k_depot, self.v_depot)
        attention_customers = multi_head_attention(self.q, self.k_customer, self.v_customer)
        attention_combine = attention_nodes + attention_depots + attention_customers
        # shape: (batch_size, mt_size, head_num * qkv_dim)
        score = self.multi_head_combine(attention_combine)
        # shape: (batch_size, mt_size, embedding_dim)
        score_nodes = torch.matmul(score, self.nodes_key)
        # shape: (batch_size, mt_size, node_size)
        sqrt_embedding_dim = self.embedding_dim ** (1 / 2)
        score_scaled = score_nodes / sqrt_embedding_dim
        # shape: (batch_size, mt_size, node_size)
        score_clipped = self.clip * torch.tanh(score_scaled)
        score_masked = score_clipped + mask
        prob = F.softmax(score_masked, dim=2)
        # shape: (batch_size, mt_size, node_size)
        return prob


class Norm(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.norm = nn.InstanceNorm1d(self.embedding_dim, affine=True, track_running_stats=False)

    def forward(self, input1, input2):
        # shape: (batch_size, node_size, embedding_dim)
        input_added = input1 + input2
        # shape: (batch_size, node_size, embedding_dim)
        input_transposed = input_added.transpose(1, 2)
        # shape: (batch_size, embedding_dim, node_size)
        input_normed = self.norm(input_transposed)
        # shape: (batch_size, embedding_dim, node_size)
        output_transposed = input_normed.transpose(1, 2)
        # shape: (batch_size, node_size, embedding_dim)
        return output_transposed


class FF(nn.Module):
    def __init__(self, **model_params):
        super().__init__()
        self.embedding_dim = model_params['embedding_dim']
        self.ff_hidden_dim = model_params['ff_hidden_dim']
        self.W1 = nn.Linear(self.embedding_dim, self.ff_hidden_dim)
        self.W2 = nn.Linear(self.ff_hidden_dim, self.embedding_dim)

    def forward(self, input1):
        # shape: (batch_size, node_size, embedding_dim)
        return self.W2(F.relu(self.W1(input1)))


def get_encoding(encoded_nodes, state):
    # encoded_customers shape: (batch_size, node_size, embedding_dim)
    # index_to_pick shape: (batch_size, mt_size)
    index_to_pick = state.current_node
    batch_size = index_to_pick.size(0)
    mt_size = index_to_pick.size(1)
    embedding_dim = encoded_nodes.size(2)
    index_to_gather = index_to_pick[:, :, None].expand(batch_size, mt_size, embedding_dim)
    # shape: (batch_size, mt_size, embedding_dim)
    picked_customers = encoded_nodes.gather(dim=1, index=index_to_gather)
    # shape: (batch_size, mt_size, embedding_dim)
    return picked_customers


def multi_head_qkv(qkv, head_num):
    # shape: (batch_size, n, embedding_dim) : n can be 1 or node_size
    batch_size = qkv.size(0)
    n = qkv.size(1)
    qkv_multi_head = qkv.reshape(batch_size, n, head_num, -1)
    qkv_transposed = qkv_multi_head.transpose(1, 2)
    # shape: (batch_size, head_num, n, key_dim)
    return qkv_transposed


def multi_head_attention(q, k, v, rank2_mask=None, rank3_mask=None):
    # q shape: (batch_size, head_num, n, key_dim)
    # k,v shape: (batch_size, head_num, node_size, key_dim)
    # rank2_mask shape: (batch_size, node_size)
    # rank3_mask shape: (batch_size, group, node_size)
    batch_size = q.size(0)
    head_num = q.size(1)
    n = q.size(2)
    key_dim = q.size(3)
    depot_customer_size = k.size(2)
    score = torch.matmul(q, k.transpose(2, 3))
    # shape :(batch_size, head_num, n, node_size)
    score_scaled = score / torch.sqrt(torch.tensor(key_dim, dtype=torch.float))
    if rank2_mask is not None:
        score_scaled = score_scaled + rank2_mask[:, None, None, :].expand(batch_size, head_num, n, depot_customer_size)
    if rank3_mask is not None:
        score_scaled = score_scaled + rank3_mask[:, None, :, :].expand(batch_size, head_num, n, depot_customer_size)
    weights = nn.Softmax(dim=3)(score_scaled)
    # shape: (batch_size, head_num, n, node_size)
    out = torch.matmul(weights, v)
    # shape: (batch_size, head_num. n, key_dim)
    out_transposed = out.transpose(1, 2)
    # shape: (batch_size, n, head_num, key_dim)
    out_concat = out_transposed.reshape(batch_size, n, head_num * key_dim)
    # shape: (batch_size, n, head_num * key_dim)
    return out_concat
