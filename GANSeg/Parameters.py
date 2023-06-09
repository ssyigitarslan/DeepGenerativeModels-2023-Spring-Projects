
class Params:
    def __init__(self):
        self.latent_dim = 256
        self.cluster_number = 8
        self.n_per_kp = 4
        self.batch_size = 16
        self.lr_gen = 1e-4 
        self.lr_disc = 4e-4 
        self.con_penalty_coef = 10 
        self.area_penalty_coef =1
        self.num_workers =0
        self.data_root = ''
        self.class_name = 'celeba_wild'
        self.image_size = 128
        self.embedding_dim = 128