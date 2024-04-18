from modeling.modeling_encoder import TextEncoder, MODEL_NAME_TO_CLASS
from utils.data_utils import *
from utils.layers import *
import torch.nn.functional as F


# 实现了一个多层消息传递神经网络用于图神经网络中的节点特征学习。
class QAGNN_Message_Passing(nn.Module):
    # QAGNN 消息传递模型类的定义

    # 初始化函数，定义模型结构和参数
    def __init__(self, args, k, n_ntype, n_etype, input_size, hidden_size, output_size, dropout=0.1):
        super().__init__()

        # 确保输入特征维度与输出特征维度相同
        assert input_size == output_size

        # 保存模型参数和节点类型、边类型的数量
        self.args = args
        self.n_ntype = n_ntype
        self.n_etype = n_etype

        # 确保输入特征维度与隐藏层特征维度相同
        assert input_size == hidden_size
        self.hidden_size = hidden_size

        # 用于节点类型嵌入的线性层，将节点类型映射到隐藏层的一半维度
        self.emb_node_type = nn.Linear(self.n_ntype, hidden_size//2)

        # 指定基础函数类型，用于边类型嵌入
        self.basis_f = 'sin'  # ['id', 'linact', 'sin', 'none']

        # 用于边类型嵌入的线性层
        if self.basis_f in ['id']:
            self.emb_score = nn.Linear(1, hidden_size//2)
        elif self.basis_f in ['linact']:
            self.B_lin = nn.Linear(1, hidden_size//2)
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)
        elif self.basis_f in ['sin']:
            self.emb_score = nn.Linear(hidden_size//2, hidden_size//2)

        # 边的编码器，用于将边的类型和头尾节点的类型进行编码
        self.edge_encoder = torch.nn.Sequential(
            torch.nn.Linear(n_etype + 1 + n_ntype * 2, hidden_size),
            torch.nn.BatchNorm1d(hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size)
        )

        # 创建 k 个 GATConvE 层，每个层都是一个 GATConvE 对象，用于执行消息传递操作
        self.k = k
        self.gnn_layers = nn.ModuleList([GATConvE(args, hidden_size, n_ntype, n_etype, self.edge_encoder) for _ in range(k)])

        # 输出层的线性变换，分别对输入特征和隐藏层特征进行线性变换
        self.Vh = nn.Linear(input_size, output_size)
        self.Vx = nn.Linear(hidden_size, output_size)

        # 激活函数和 Dropout 层
        self.activation = GELU()
        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

    # 执行多层消息传递操作
    def mp_helper(self, _X, edge_index, edge_type, _node_type, _node_feature_extra):
        for _ in range(self.k):
            _X = self.gnn_layers[_](_X, edge_index, edge_type, _node_type, _node_feature_extra)
            _X = self.activation(_X)
            _X = F.dropout(_X, self.dropout_rate, training=self.training)
        return _X

    # 前向传播函数
    def forward(self, H, A, node_type, node_score, cache_output=False):
        """
        H: tensor of shape (batch_size, n_node, d_node)
            node features from the previous layer
        A: (edge_index, edge_type)
            edge_index: tensor of shape (2, total_E)
            edge_type: tensor of shape (total_E,)
            where total_E is for the batched graph
        node_type: long tensor of shape (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_score: tensor of shape (batch_size, n_node, 1)
        """

        # 获取 batch_size 和节点数量
        _batch_size, _n_nodes = node_type.size()

        # 对节点类型进行独热编码并进行嵌入处理
        T = make_one_hot(node_type.view(-1).contiguous(), self.n_ntype).view(_batch_size, _n_nodes, self.n_ntype)
        node_type_emb = self.activation(self.emb_node_type(T))  # [batch_size, n_node, dim/2]

        # 根据不同的嵌入函数对节点得分进行处理
        if self.basis_f == 'sin':
            js = torch.arange(self.hidden_size//2).unsqueeze(0).unsqueeze(0).float().to(node_type.device)
            js = torch.pow(1.1, js)
            B = torch.sin(js * node_score)
            node_score_emb = self.activation(self.emb_score(B))
        elif self.basis_f == 'id':
            B = node_score
            node_score_emb = self.activation(self.emb_score(B))
        elif self.basis_f == 'linact':
            B = self.activation(self.B_lin(node_score))
            node_score_emb = self.activation(self.emb_score(B))

        # 将节点特征和额外的节点特征拼接在一起
        X = H
        edge_index, edge_type = A
        _X = X.view(-1, X.size(2)).contiguous()
        _node_type = node_type.view(-1).contiguous()
        _node_feature_extra = torch.cat([node_type_emb, node_score_emb], dim=2).view(_node_type.size(0), -1).contiguous()

        _X = self.mp_helper(_X, edge_index, edge_type, _node_type, _node_feature_extra)

        X = _X.view(node_type.size(0), node_type.size(1), -1)  # [batch_size, n_node, dim]

        # 通过线性层和激活函数处理节点特征，并返回结果
        output = self.activation(self.Vh(H) + self.Vx(X))
        output = self.dropout(output)

        return output



# 实现了一个结合了图神经网络和注意力池化的多层感知机网络用于文本和图数据的融合
class QAGNN(nn.Module):

    # args: 模型参数。
    # k: 消息传递的层数。
    # n_ntype: 节点类型的数量。
    # n_etype: 边类型的数量。
    # sent_dim: 输入文本特征的维度。
    # n_concept: 概念数量。
    # concept_dim: 概念嵌入的维度。
    # concept_in_dim: 输入概念特征的维度。
    # n_attention_head: 注意力头的数量。
    # fc_dim: 全连接层的隐藏层维度。
    # n_fc_layer: 全连接层的层数。
    # p_emb: 嵌入层的dropout概率。
    # p_gnn: GNN层的dropout概率。
    # p_fc: 全连接层的dropout概率。
    # pretrained_concept_emb: 预训练的概念嵌入。
    # freeze_ent_emb: 是否冻结实体嵌入。
    # init_range: 参数初始化范围。
    # encoder_config: 编码器配置参数。
    def __init__(self, args, k, n_ntype, n_etype, sent_dim,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.02):


        super().__init__()
        self.init_range = init_range

        # 概念嵌入层，将概念 ID 转换为概念嵌入向量
        self.concept_emb = CustomizedEmbedding(
            concept_num=n_concept, concept_out_dim=concept_dim,
            use_contextualized=False, concept_in_dim=concept_in_dim,
            pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb
        )

        # 线性层，用于将句子特征转换为节点特征
        self.svec2nvec = nn.Linear(sent_dim, concept_dim)

        self.concept_dim = concept_dim

        # 激活函数
        self.activation = GELU()

        # 图神经网络层
        self.gnn = QAGNN_Message_Passing(args, k=k, n_ntype=n_ntype, n_etype=n_etype,
                                         input_size=concept_dim, hidden_size=concept_dim, output_size=concept_dim,
                                         dropout=p_gnn)

        # 注意力池化层
        self.pooler = MultiheadAttPoolLayer(n_attention_head, sent_dim, concept_dim)

        # 全连接层
        self.fc = MLP(concept_dim + sent_dim + concept_dim, fc_dim, 1, n_fc_layer, p_fc, layer_norm=True)

        # Dropout 层
        self.dropout_e = nn.Dropout(p_emb)
        self.dropout_fc = nn.Dropout(p_fc)

        # 参数初始化
        if init_range > 0:
            self.apply(self._init_weights)


    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.init_range)
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)



    # sent_vecs: 文本特征向量，维度为(batch_size, num_choice, sent_dim)。
    # concept_ids: 概念ID，维度为(batch_size, num_choice, n_node)。
    # node_type_ids: 节点类型ID，维度为(batch_size, num_choice, n_node)。
    # node_scores: 节点得分，维度为(batch_size, num_choice, 1)。
    # adj_lengths: 图中每个例子的有效节点长度，维度为(batch_size, )。
    # adj: 图的邻接矩阵，包含edge_index和edge_type。
    # emb_data: 嵌入数据，默认为None。
    # cache_output: 是否缓存输出，默认为False。
    def forward(self, sent_vecs, concept_ids, node_type_ids, node_scores, adj_lengths, adj, emb_data=None, cache_output=False):
        """
        sent_vecs: (batch_size, dim_sent)
        concept_ids: (batch_size, n_node)
        adj: edge_index, edge_type
        adj_lengths: (batch_size,)
        node_type_ids: (batch_size, n_node)
            0 == question entity; 1 == answer choice entity; 2 == other node; 3 == context node
        node_scores: (batch_size, n_node, 1)

        returns: (batch_size, 1)
        """

        # 使用svec2nvec将句子向量转换为节点向量。
        gnn_input0 = self.activation(self.svec2nvec(sent_vecs)).unsqueeze(1) #(batch_size, 1, dim_node)

        # 使用 concept_emb 将概念 ID 转换为概念嵌入向量。
        gnn_input1 = self.concept_emb(concept_ids[:, 1:]-1, emb_data) #(batch_size, n_node-1, dim_node)

        # 使用 gnn 执行消息传递操作，更新节点特征。
        gnn_input1 = gnn_input1.to(node_type_ids.device)
        gnn_input = self.dropout_e(torch.cat([gnn_input0, gnn_input1], dim=1)) #(batch_size, n_node, dim_node)


        #Normalize node sore (use norm from Z)
        _mask = (torch.arange(node_scores.size(1), device=node_scores.device) < adj_lengths.unsqueeze(1)).float() #0 means masked out #[batch_size, n_node]
        node_scores = -node_scores
        node_scores = node_scores - node_scores[:, 0:1, :] #[batch_size, n_node, 1]
        node_scores = node_scores.squeeze(2) #[batch_size, n_node]
        node_scores = node_scores * _mask
        mean_norm  = (torch.abs(node_scores)).sum(dim=1) / adj_lengths  #[batch_size, ]
        node_scores = node_scores / (mean_norm.unsqueeze(1) + 1e-05) #[batch_size, n_node]
        node_scores = node_scores.unsqueeze(2) #[batch_size, n_node, 1]


        gnn_output = self.gnn(gnn_input, adj, node_type_ids, node_scores)

        Z_vecs = gnn_output[:,0]   #(batch_size, dim_node)

        mask = torch.arange(node_type_ids.size(1), device=node_type_ids.device) >= adj_lengths.unsqueeze(1) #1 means masked out

        mask = mask | (node_type_ids == 3) #pool over all KG nodes
        mask[mask.all(1), 0] = 0  # a temporary solution to avoid zero node


        # 使用 pooler 对节点特征进行注意力池化。
        sent_vecs_for_pooler = sent_vecs
        graph_vecs, pool_attn = self.pooler(sent_vecs_for_pooler, gnn_output, mask)

        if cache_output:
            self.concept_ids = concept_ids
            self.adj = adj
            self.pool_attn = pool_attn


        # 将池化后的节点特征与句子向量和节点嵌入向量拼接。
        concat = self.dropout_fc(torch.cat((graph_vecs, sent_vecs, Z_vecs), 1))

        # 使用 fc 对拼接后的特征进行多层感知机处理。
        logits = self.fc(concat)
        return logits, pool_attn


# 实现了对文本和图数据的融合处理。
class LM_QAGNN(nn.Module):
    def __init__(self, args, model_name, k, n_ntype, n_etype,
                 n_concept, concept_dim, concept_in_dim, n_attention_head,
                 fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                 pretrained_concept_emb=None, freeze_ent_emb=True,
                 init_range=0.0, encoder_config={}):
        super().__init__()

        # 使用 TextEncoder 将文本特征编码成向量表示
        self.encoder = TextEncoder(model_name, **encoder_config)

        # 使用 QAGNN 构建问题-答案图神经网络模型
        self.decoder = QAGNN(args, k, n_ntype, n_etype, self.encoder.sent_dim,
                                        n_concept, concept_dim, concept_in_dim, n_attention_head,
                                        fc_dim, n_fc_layer, p_emb, p_gnn, p_fc,
                                        pretrained_concept_emb=pretrained_concept_emb, freeze_ent_emb=freeze_ent_emb,
                                        init_range=init_range)


    def forward(self, *inputs, layer_id=-1, cache_output=False, detail=False):
        """
        sent_vecs: (batch_size, num_choice, d_sent)    -> (batch_size * num_choice, d_sent)
        concept_ids: (batch_size, num_choice, n_node)  -> (batch_size * num_choice, n_node)
        node_type_ids: (batch_size, num_choice, n_node) -> (batch_size * num_choice, n_node)
        adj_lengths: (batch_size, num_choice)          -> (batch_size * num_choice, )
        adj -> edge_index, edge_type
            edge_index: list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(2, E(variable))
                                                         -> (2, total E)
            edge_type:  list of (batch_size, num_choice) -> list of (batch_size * num_choice, ); each entry is torch.tensor(E(variable), )
                                                         -> (total E, )
        returns: (batch_size, 1)
        """

        # 打印输入参数
        print("inputs: ", inputs)
        bs, nc = inputs[0].size(0), inputs[0].size(1)

        # 合并批次维度和选项数量维度
        edge_index_orig, edge_type_orig = inputs[-2:]
        a = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[:-6]]
        b = [x.view(x.size(0) * x.size(1), *x.size()[2:]) for x in inputs[-6:-2]]
        c = [sum(x,[]) for x in inputs[-2:]]
        _inputs = a + b + c
        *lm_inputs, concept_ids, node_type_ids, node_scores, adj_lengths, edge_index, edge_type = _inputs
        edge_index, edge_type = self.batch_graph(edge_index, edge_type, concept_ids.size(1))
        adj = (edge_index.to(node_type_ids.device), edge_type.to(node_type_ids.device)) #edge_index: [2, total_E]   edge_type: [total_E, ]

        # 编码文本特征并获取模型输出
        sent_vecs, all_hidden_states = self.encoder(*lm_inputs, layer_id=layer_id)
        logits, attn = self.decoder(sent_vecs.to(node_type_ids.device),
                                    concept_ids,
                                    node_type_ids, node_scores, adj_lengths, adj,
                                    emb_data=None, cache_output=cache_output)

        # 将输出的logits形状转换为(batch_size, num_choice)
        logits = logits.view(bs, nc)

        # 如果不需要详细输出，则返回logits和注意力权重
        if not detail:
            return logits, attn
        else:
            # 如果需要详细输出，则返回logits、注意力权重、概念ID、节点类型ID、原始的边索引和边类型
            return logits, attn, concept_ids.view(bs, nc, -1), node_type_ids.view(bs, nc, -1), edge_index_orig, edge_type_orig


    def batch_graph(self, edge_index_init, edge_type_init, n_nodes):
        #edge_index_init: list of (n_examples, ). each entry is torch.tensor(2, E)
        #edge_type_init:  list of (n_examples, ). each entry is torch.tensor(E, )
        n_examples = len(edge_index_init)
        edge_index = [edge_index_init[_i_] + _i_ * n_nodes for _i_ in range(n_examples)]
        edge_index = torch.cat(edge_index, dim=1) #[2, total_E]
        edge_type = torch.cat(edge_type_init, dim=0) #[total_E, ]
        return edge_index, edge_type


# 用于加载训练、开发和测试数据，并提供数据生成器以便模型进行训练、评估和测试。
class LM_QAGNN_DataLoader(object):
    def __init__(self, args, train_statement_path, train_adj_path,
                 dev_statement_path, dev_adj_path,
                 test_statement_path, test_adj_path,
                 batch_size, eval_batch_size, device, model_name, max_node_num=200, max_seq_length=128,
                 is_inhouse=False, inhouse_train_qids_path=None,
                 subsample=1.0, use_cache=True):
        super().__init__()
        self.args = args
        self.batch_size = batch_size
        self.eval_batch_size = eval_batch_size
        self.device0, self.device1 = device
        self.is_inhouse = is_inhouse

        model_type = MODEL_NAME_TO_CLASS[model_name]

        # 加载训练、开发和测试集的输入数据
        self.train_qids, self.train_labels, *self.train_encoder_data = load_input_tensors(train_statement_path, model_type, model_name, max_seq_length)
        self.dev_qids, self.dev_labels, *self.dev_encoder_data = load_input_tensors(dev_statement_path, model_type, model_name, max_seq_length)
        self.test_qids, self.test_labels, *self.test_encoder_data = load_input_tensors(test_statement_path, model_type, model_name, max_seq_length) if test_statement_path is not None else (None, None, None)

        # 加载训练、开发和测试集的图数据
        *self.train_decoder_data, self.train_adj_data = load_sparse_adj_data_with_contextnode(train_adj_path, max_node_num, num_choice, args)
        *self.dev_decoder_data, self.dev_adj_data = load_sparse_adj_data_with_contextnode(dev_adj_path, max_node_num, num_choice, args)
        *self.test_decoder_data, self.test_adj_data = load_sparse_adj_data_with_contextnode(test_adj_path, max_node_num, num_choice, args) if test_adj_path is not None else (None, None)

        # 确保数据的一致性
        assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
        assert all(len(self.dev_qids) == len(self.dev_adj_data[0]) == x.size(0) for x in [self.dev_labels] + self.dev_encoder_data + self.dev_decoder_data)
        if test_statement_path is not None:
            assert all(len(self.test_qids) == len(self.test_adj_data[0]) == x.size(0) for x in [self.test_labels] + self.test_encoder_data + self.test_decoder_data)

        # 若为内部训练，则加载内部训练数据
        if self.is_inhouse:
            with open(inhouse_train_qids_path, 'r') as fin:
                inhouse_qids = set(line.strip() for line in fin)
            self.inhouse_train_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid in inhouse_qids])
            self.inhouse_test_indexes = torch.tensor([i for i, qid in enumerate(self.train_qids) if qid not in inhouse_qids])

        # 对数据进行子采样
        assert 0. < subsample <= 1.
        if subsample < 1.:
            n_train = int(self.train_size() * subsample)
            assert n_train > 0
            if self.is_inhouse:
                self.inhouse_train_indexes = self.inhouse_train_indexes[:n_train]
            else:
                self.train_qids = self.train_qids[:n_train]
                self.train_labels = self.train_labels[:n_train]
                self.train_encoder_data = [x[:n_train] for x in self.train_encoder_data]
                self.train_decoder_data = [x[:n_train] for x in self.train_decoder_data]
                self.train_adj_data = self.train_adj_data[:n_train]
                assert all(len(self.train_qids) == len(self.train_adj_data[0]) == x.size(0) for x in [self.train_labels] + self.train_encoder_data + self.train_decoder_data)
            assert self.train_size() == n_train

    # 返回训练集大小
    def train_size(self):
        return self.inhouse_train_indexes.size(0) if self.is_inhouse else len(self.train_qids)

    # 返回开发集大小
    def dev_size(self):
        return len(self.dev_qids)

    # 返回测试集大小
    def test_size(self):
        if self.is_inhouse:
            return self.inhouse_test_indexes.size(0)
        else:
            return len(self.test_qids) if self.test_qids is not None else 0

    # 返回训练集的数据生成器
    def train(self):
        if self.is_inhouse:
            n_train = self.inhouse_train_indexes.size(0)
            train_indexes = self.inhouse_train_indexes[torch.randperm(n_train)]
        else:
            train_indexes = torch.randperm(len(self.train_qids))
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'train', self.device0, self.device1, self.batch_size, train_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    # 返回训练集评估数据生成器
    def train_eval(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.train_qids)), self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)

    # 返回开发集的数据生成器
    def dev(self):
        return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.dev_qids)), self.dev_qids, self.dev_labels, tensors0=self.dev_encoder_data, tensors1=self.dev_decoder_data, adj_data=self.dev_adj_data)

    # 返回测试集的数据生成器
    def test(self):
        if self.is_inhouse:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, self.inhouse_test_indexes, self.train_qids, self.train_labels, tensors0=self.train_encoder_data, tensors1=self.train_decoder_data, adj_data=self.train_adj_data)
        else:
            return MultiGPUSparseAdjDataBatchGenerator(self.args, 'eval', self.device0, self.device1, self.eval_batch_size, torch.arange(len(self.test_qids)), self.test_qids, self.test_labels, tensors0=self.test_encoder_data, tensors1=self.test_decoder_data, adj_data=self.test_adj_data)



###############################################################################
############################### GNN architecture ##############################
###############################################################################

from torch.autograd import Variable
def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.
    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.FloatTensor(labels.size(0), C).zero_().to(labels.device)
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target



from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, softmax
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter
from torch_geometric.nn.inits import glorot, zeros



# 基于GAT（Graph Attention Network）的消息传递层，其中GATConvE类继承自MessagePassing类。
class GATConvE(MessagePassing):
    """
    Args:
        emb_dim (int): GNN隐藏状态的维度
        n_ntype (int): 节点类型的数量（例如4）
        n_etype (int): 边关系类型的数量（例如38）
    """
    def __init__(self, args, emb_dim, n_ntype, n_etype, edge_encoder, head_count=4, aggr="add"):
        super(GATConvE, self).__init__(aggr=aggr)
        self.args = args

        assert emb_dim % 2 == 0
        self.emb_dim = emb_dim

        self.n_ntype = n_ntype; self.n_etype = n_etype
        self.edge_encoder = edge_encoder

        # 注意力机制
        self.head_count = head_count
        assert emb_dim % head_count == 0
        self.dim_per_head = emb_dim // head_count
        self.linear_key = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_msg = nn.Linear(3*emb_dim, head_count * self.dim_per_head)
        self.linear_query = nn.Linear(2*emb_dim, head_count * self.dim_per_head)

        self._alpha = None

        # 最终的MLP
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, emb_dim), torch.nn.BatchNorm1d(emb_dim), torch.nn.ReLU(), torch.nn.Linear(emb_dim, emb_dim))


    def forward(self, x, edge_index, edge_type, node_type, node_feature_extra, return_attention_weights=False):
        # x: [N, emb_dim] 节点特征
        # edge_index: [2, E] 边索引
        # edge_type [E,] -> edge_attr: [E, 39] / self_edge_attr: [N, 39] 边类型
        # node_type [N,] -> headtail_attr [E, 8(=4+4)] / self_headtail_attr: [N, 8] 节点类型
        # node_feature_extra [N, dim] 节点额外特征

        # 准备边特征
        edge_vec = make_one_hot(edge_type, self.n_etype + 1)  # [E, 39]
        self_edge_vec = torch.zeros(x.size(0), self.n_etype + 1).to(edge_vec.device)
        self_edge_vec[:, self.n_etype] = 1

        head_type = node_type[edge_index[0]]  # [E,] 头部=源节点
        tail_type = node_type[edge_index[1]]  # [E,] 尾部=目标节点
        head_vec = make_one_hot(head_type, self.n_ntype)  # [E,4]
        tail_vec = make_one_hot(tail_type, self.n_ntype)  # [E,4]
        headtail_vec = torch.cat([head_vec, tail_vec], dim=1)  # [E,8]
        self_head_vec = make_one_hot(node_type, self.n_ntype)  # [N,4]
        self_headtail_vec = torch.cat([self_head_vec, self_head_vec], dim=1)  # [N,8]

        edge_vec = torch.cat([edge_vec, self_edge_vec], dim=0)  # [E+N, ?]
        headtail_vec = torch.cat([headtail_vec, self_headtail_vec], dim=0)  # [E+N, ?]
        edge_embeddings = self.edge_encoder(torch.cat([edge_vec, headtail_vec], dim=1))  # [E+N, emb_dim]

        # 添加自环到边索引
        loop_index = torch.arange(0, x.size(0), dtype=torch.long, device=edge_index.device)
        loop_index = loop_index.unsqueeze(0).repeat(2, 1)
        edge_index = torch.cat([edge_index, loop_index], dim=1)  # [2, E+N]

        x = torch.cat([x, node_feature_extra], dim=1)
        x = (x, x)
        aggr_out = self.propagate(edge_index, x=x, edge_attr=edge_embeddings)  # [N, emb_dim]
        out = self.mlp(aggr_out)

        alpha = self._alpha
        self._alpha = None

        if return_attention_weights:
            assert alpha is not None
            return out, (edge_index, alpha)
        else:
            return out


    def message(self, edge_index, x_i, x_j, edge_attr):  # i: 目标节点, j: 源节点
        assert len(edge_attr.size()) == 2
        assert edge_attr.size(1) == self.emb_dim
        assert x_i.size(1) == x_j.size(1) == 2*self.emb_dim
        assert x_i.size(0) == x_j.size(0) == edge_attr.size(0) == edge_index.size(1)

        key = self.linear_key(torch.cat([x_i, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        msg = self.linear_msg(torch.cat([x_j, edge_attr], dim=1)).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]
        query = self.linear_query(x_j).view(-1, self.head_count, self.dim_per_head)  # [E, heads, _dim]

        query = query / math.sqrt(self.dim_per_head)
        scores = (query * key).sum(dim=2)  # [E, heads]
        src_node_index = edge_index[0]  # [E,]
        alpha = softmax(scores, src_node_index)  # [E, heads] 按源节点分组
        self._alpha = alpha

        # 根据源节点的出度调整
        E = edge_index.size(1)            # 边数
        N = int(src_node_index.max()) + 1  # 节点数
        ones = torch.full((E,), 1.0, dtype=torch.float).to(edge_index.device)
        src_node_edge_count = scatter(ones, src_node_index, dim=0, dim_size=N, reduce='sum')[src_node_index]  # [E,]
        assert len(src_node_edge_count.size()) == 1 and len(src_node_edge_count) == E
        alpha = alpha * src_node_edge_count.unsqueeze(1)  # [E, heads]

        out = msg * alpha.view(-1, self.head_count, 1)  # [E, heads, _dim]
        return out.view(-1, self.head_count * self.dim_per_head)  # [E, emb_dim]