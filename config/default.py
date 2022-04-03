import math

from yacs.config import CfgNode as CN

_c = CN()
_c.model_dim = 2048   # should be adjusted for different dataset
_c.max_word_num = 20
_c.max_frame_num = 64
_c.chunk_num = 16  # 8 for TACoS and Charades， 16 for anet
_c.core_dim = _c.model_dim // _c.chunk_num

# GNN Config
_c.GNN = CN()
_c.GNN.RGCN = CN(new_allowed=True)
_c.GNN.RGCN.in_channels = _c.core_dim
_c.GNN.RGCN.out_channels = _c.core_dim
_c.GNN.RGCN.num_relations = 2

_c.GNN.GAT = CN(new_allowed=True)
_c.GNN.GAT.in_channels = _c.core_dim
_c.GNN.GAT.out_channels = _c.core_dim
_c.GNN.GAT.concat = False
_c.GNN.GAT.heads = 1

_c.GNN.GEN = CN(new_allowed=True)
_c.GNN.GEN.in_channels = _c.core_dim
_c.GNN.GEN.out_channels = _c.core_dim
_c.GNN.GEN.aggr = 'softmax'
_c.GNN.GEN.t = 1.0
_c.GNN.GEN.learn_t = True
_c.GNN.GEN.num_layers = 2
_c.GNN.GEN.norm = 'layer'

_c.competitive_tree = CN()
_c.competitive_tree.hide_ratio = 0
_c.competitive_tree.dropout = 0.4  # 0.2
_c.competitive_tree.model_dim = _c.core_dim
_c.competitive_tree.max_element_num = _c.max_frame_num
_c.competitive_tree.compress_method = "CrossGate"  # "CrossGate"
_c.competitive_tree.task_num = 3
_c.competitive_tree.attn_size = 1
_c.competitive_tree.use_mounted_feature = False
_c.competitive_tree.use_root_feature = False
_c.competitive_tree.use_positional_encoding = True
_c.competitive_tree.graph_config = CN()
_c.competitive_tree.graph_config.layer_num = 4
_c.competitive_tree.graph_config.name = "GEN"
_c.competitive_tree.graph_config.model_dim = _c.core_dim
_c.competitive_tree.graph_config.feedforward_dim = _c.core_dim * 4
_c.competitive_tree.graph_config.edge_type = "Categorical"
# _c.competitive_tree.graph_config.edge_type = "Categorical"
# edge_type is "Categorical" or "Hierarchical"
_c.competitive_tree.graph_config.dropout = 0.5
_c.competitive_tree.graph_config.gnn_config = _c.GNN[_c.competitive_tree.graph_config.name]

# _c.ensemble_tree = CN()
# _c.ensemble_tree.hide_ratio = 0
# _c.ensemble_tree.model_dim = _c.core_dim
# _c.ensemble_tree.max_element_num = _c.max_frame_num
# _c.ensemble_tree.compress_method = "Conv"
# _c.ensemble_tree.task_num = 2
# _c.ensemble_tree.attn_size = 1
# _c.ensemble_tree.use_mounted_feature = True
# _c.ensemble_tree.use_root_feature = False
# _c.ensemble_tree.use_positional_encoding = True
# _c.ensemble_tree.graph_config = CN()
# _c.ensemble_tree.graph_config.layer_num = 3
# _c.ensemble_tree.graph_config.name = "GAT"
# _c.ensemble_tree.graph_config.model_dim = _c.core_dim
# _c.ensemble_tree.graph_config.feedforward_dim = _c.core_dim * 4
# _c.ensemble_tree.graph_config.edge_type = "Hierarchical"
# # edge_type is "Category" or "Hierarchical"
# _c.ensemble_tree.graph_config.gnn_config = _c.GNN[_c.ensemble_tree.graph_config.name]

# Model Config
_c.model = CN()
_c.model.name = "MomentRetrievalModel"
_c.model.model_dim = _c.model_dim
_c.model.chunk_num = _c.chunk_num
_c.model.use_negative = False
_c.model.text_config = CN()
_c.model.text_config.input_dim = 300
_c.model.text_config.hidden_dim = _c.model_dim
_c.model.video_config = CN()
_c.model.video_config.input_dim = 500
_c.model.video_config.hidden_dim = _c.model_dim
_c.model.video_config.kernel_size = [1]
_c.model.video_config.stride = [1]
_c.model.video_config.padding = [0]

_c.model.tree_config = _c.competitive_tree


# Train Config
_c.train = CN()
_c.train.batch_size = 64
_c.train.max_epoch = 16
_c.train.display_interval = 50
_c.train.saved_path = "checkpoints/anet_other"

# _c.val = CN()
# _c.val.batch_size = 96

# Test Config
_c.test = CN()
_c.test.batch_size = 96
_c.test.type_list = ["moment_retrieval"]
_c.test.args_list = [{"top_n": 5, "thresh": 0.5, "by_frame": False, "display_interval": 100}]

# Dataset Config
_c.dataset = CN()
# _c.dataset.name = "CharadesSTA"
# _c.dataset.feature_path = "../data/Charades-STA/Charades-pca-500"
# _c.dataset.vocab_path = "data/charades_sta/glove_model.bin"
# _c.dataset.data_path_template = "data/charades_sta/{}.json"
_c.dataset.name = "AnetCaption"
_c.dataset.feature_path = "../../data/activity-c3d"
_c.dataset.vocab_path = "data/activitynet/glove_model.bin"
_c.dataset.data_path_template = "data/activitynet/{}_data.json"
_c.dataset.max_frame_num = _c.max_frame_num
_c.dataset.max_word_num = _c.max_word_num
_c.dataset.frame_dim = _c.model.video_config.input_dim
_c.dataset.word_dim = _c.model.text_config.input_dim

# Optimizer Config
_c.optimizer = CN()
_c.optimizer.lr = 1e-3
_c.optimizer.warmup_updates = 800  # 400 for anet
_c.optimizer.warmup_init_lr = 1e-7
_c.optimizer.weight_decay = 1e-4
_c.optimizer.loss_config = CN()
_c.optimizer.loss_config.boundary = 1
_c.optimizer.loss_config.inside = 6
_c.optimizer.loss_config.norm_bias = 1


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _c.clone()
