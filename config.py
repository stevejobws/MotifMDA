class Config(object):
    def __init__(self):
        self.data_path = 'data'
        self.save_path = 'result'
        self.epoch = 150
        self.SEED = 0
        self.FOLD = 1
        self.m_size = 495  # hmddv2 miRNA numbers
        self.d_size = 383
        self.Threshold = 0.5
        self.K = 7  # sample size  实例采样数量
        self.in_dim = 512
        self.out1_dim = 256
        self.out2_dim = 128
        self.out3_dim = 64
        self.patience = 60
        self.dropout_p = 0.5
        self.alpha = 0.2  # for leakyrelu
        self.torch_seed = 0
