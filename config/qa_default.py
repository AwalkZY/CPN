from yacs.config import CfgNode as CN

_c = CN()
_c.model_dim = 2048   # should be adjusted for different dataset
_c.max_word_num = 20
_c.max_frame_num = 16
_c.chunk_num = 2  # 8 for TACoS and Charades， 16 for anet
_c.core_dim = _c.model_dim // _c.chunk_num

# GNN Config
_c.GNN = CN()
# _c.GNN.RGCN = CN(new_allowed=True)
# _c.GNN.RGCN.in_channels = _c.core_dim
# _c.GNN.RGCN.out_channels = _c.core_dim
# _c.GNN.RGCN.num_relations = 16
# _c.GNN.RGCN.num_bases = 4

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
_c.GNN.GEN.num_layers = 1
_c.GNN.GEN.norm = 'layer'

_c.ensemble_tree = CN()
_c.ensemble_tree.hide_ratio = 0
_c.ensemble_tree.dropout = 0.4  # 0.2
_c.ensemble_tree.model_dim = _c.core_dim
_c.ensemble_tree.max_element_num = _c.max_frame_num
_c.ensemble_tree.compress_method = "Pool"
_c.ensemble_tree.task_num = 1
_c.ensemble_tree.attn_size = 1
_c.ensemble_tree.class_num = 1000
_c.ensemble_tree.use_mounted_feature = False
_c.ensemble_tree.use_root_feature = False
_c.ensemble_tree.use_positional_encoding = True
_c.ensemble_tree.graph_config = CN()
_c.ensemble_tree.graph_config.layer_num = 1
_c.ensemble_tree.graph_config.name = "GEN"
_c.ensemble_tree.graph_config.model_dim = _c.core_dim
_c.ensemble_tree.graph_config.feedforward_dim = _c.core_dim * 4
_c.ensemble_tree.graph_config.edge_type = "Hierarchical"
# edge_type is "Category" or "Hierarchical"
_c.ensemble_tree.graph_config.dropout = 0.5  # 0.4
_c.ensemble_tree.graph_config.gnn_config = _c.GNN[_c.ensemble_tree.graph_config.name]


# Model Config
_c.model = CN()
_c.model.name = "QuestionAnswerModel"
_c.model.model_dim = _c.model_dim
_c.model.chunk_num = _c.chunk_num
_c.model.use_negative = False
_c.model.text_config = CN()
_c.model.text_config.input_dim = 300
_c.model.text_config.hidden_dim = _c.model_dim
_c.model.video_config = CN()
_c.model.video_config.input_dim = 8192
_c.model.video_config.frame_dim = 4096
_c.model.video_config.motion_dim = 4096
_c.model.video_config.hidden_dim = _c.model_dim
_c.model.video_config.kernel_size = [1]
_c.model.video_config.stride = [1]
_c.model.video_config.padding = [0]

_c.model.tree_config = _c.ensemble_tree


# Train Config
_c.train = CN()
_c.train.batch_size = 32
_c.train.max_epoch = 25
_c.train.display_interval = 50
_c.train.saved_path = "checkpoints/msvd"

# Test Config
_c.test = CN()
_c.test.batch_size = 32
_c.test.type_list = ["question_answer"]
_c.test.args_list = [{"answer_path": "/home1/zhaoyang/data/msvd/answer_set.txt", "display_interval": 20}]

# Dataset Config
_c.dataset = CN()
_c.dataset.shared_cfg = CN()
_c.dataset.shared_cfg.data_path = "../data/msvd"
_c.dataset.shared_cfg.name = "MSVDQA"
_c.dataset.shared_cfg.max_frame_num = _c.max_frame_num
# 2-6; 7-11; 12-16; 17-21
_c.dataset_train_chunk_0 = CN()
_c.dataset_train_chunk_0.min_word_num = 0
_c.dataset_train_chunk_0.max_word_num = 21
_c.dataset.train_chunk_1 = CN()
_c.dataset.train_chunk_1.min_word_num = 2
_c.dataset.train_chunk_1.max_word_num = 6
_c.dataset.train_chunk_2 = CN()
_c.dataset.train_chunk_2.min_word_num = 7
_c.dataset.train_chunk_2.max_word_num = 11
_c.dataset.train_chunk_3 = CN()
_c.dataset.train_chunk_3.min_word_num = 12
_c.dataset.train_chunk_3.max_word_num = 16
_c.dataset.train_chunk_4 = CN()
_c.dataset.train_chunk_4.min_word_num = 17
_c.dataset.train_chunk_4.max_word_num = 21
# _c.dataset.train_data_cfg = [_c.dataset.train_chunk_1, _c.dataset.train_chunk_2,
#                              _c.dataset.train_chunk_3, _c.dataset.train_chunk_4]
_c.dataset.train_data_cfg = [_c.dataset_train_chunk_0]
_c.dataset.val_data_cfg = CN()
_c.dataset.val_data_cfg.min_word_num = 0
_c.dataset.val_data_cfg.max_word_num = 21
_c.dataset.test_data_cfg = CN()
_c.dataset.test_data_cfg.min_word_num = 0
_c.dataset.test_data_cfg.max_word_num = 21

# Optimizer Config
_c.optimizer = CN()
_c.optimizer.lr = 5e-4
_c.optimizer.warmup_updates = 200
_c.optimizer.warmup_init_lr = 1e-7
_c.optimizer.weight_decay = 1e-4
_c.optimizer.loss_config = CN()


def get_cfg_defaults():
    """Get a yacs CfgNode object with default values for my_project."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    return _c.clone()
