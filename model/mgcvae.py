import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from model.model_utils import ModeKeys,rgetattr,rsetattr,CustomLR,exp_anneal,sigmoid_anneal,unpack_RNN_state,run_lstm_on_variable_length_seqs,mutual_inf_mc
from model.dynamics import SE3Integrator
from model.discrete_latent import DiscreteLatent
from model.pn import SimplePointCloudTransformer, padding, ptv3_pad_features, PointCloudTransformerDecoder
import time
import torch.distributions as td
from model.gmm_rot import GMM3D
from model.gmm6d import GMM6D
from model.ptv3 import load_miniptv3
from scipy.stats import chi2

class MultimodalGenerativeCVAE(nn.Module):
    def __init__(self,
                 hyperparams,
                 device):
        super(MultimodalGenerativeCVAE,self).__init__()
        self.hyperparams = hyperparams
        self.device = device
        self.curr_iter = 0
        # self.arch = self.hyperparams['arch']

        self.node_modules = nn.ModuleDict()

        self.min_hl = self.hyperparams['minimum_history_length']
        self.max_hl = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.pred_state = self.hyperparams['pred_state']
        self.state_length = int(np.sum([len(entity_dims) for entity_dims in self.state.values()]))

        self.pred_state_length = int(np.sum([len(entity_dims) for entity_dims in self.pred_state.values()]))
        self.create_graphical_model()

        dyn_limits = hyperparams['dynamic']['limits']
        self.dynamic = SE3Integrator(1./self.hyperparams['frequency'], dyn_limits, device)
        self.scale = 5

        # RL params
        self.goal_threshold = 0.05
        self.goal_reward = 1
        self.col_threshold = 0.05
        self.col_reward = -1
        self.discount_factor = 0.9
        self.pg_alpha = 1.0
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        
        self.critic_w = 0.5
        self.total_it = 0
        self.target_update_interval = 1
        self.tau = 0.005


        ## ood detection
        self.ood_buffer = []
        self.ood_buffer_size = 1000
        self.ood_check_steps = 1
        self.ood_steps_after = 0
        self.trans_entropy_lb = 100
        self.trans_entropy_ub = 100
        self.rot_entropy_lb = 100
        self.rot_entropy_ub = 100
        self.ood_alpha = hyperparams['ood_alpha']

        self.cfg = hyperparams.get('cfg', False)
        self.cfg_training = hyperparams.get('cfg_training', False)


    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter

    def add_submodule(self, name, model):
        self.node_modules[name] = model.to(self.device)

    def clear_submodules(self):
        self.node_modules.clear()

    def create_node_models(self):
        ############################
        #   Node History Encoder   #
        ############################
        self.add_submodule('/history_encoder',
                        model=nn.LSTM(input_size=self.state_length,
                                        hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                        batch_first=True))

        if self.hyperparams['js_encoding']:
            self.add_submodule('/js_encoder',
                            model=nn.LSTM(input_size=7,
                                        hidden_size=self.hyperparams['enc_rnn_dim_history'],
                                        batch_first=True))

        
        if self.hyperparams['grasp_encoding']:
            self.add_submodule('/grasp_encoder', 
                               model=SimplePointCloudTransformer(input_dim=6, tgt_dim=self.hyperparams['enc_rnn_dim_history'], output_dim=self.hyperparams['enc_grasp_dim']))
            self.grasp_output_dims = self.hyperparams['enc_grasp_dim']
            # self.add_submodule('/attractor', model=PointCloudTransformerDecoder(tgt_dim=self.hyperparams['enc_grasp_dim'], model_dim=self.hyperparams['dec_rnn_dim'], num_layers=1, output_dim=1))
            self.add_submodule('/attractor', model=nn.Linear(self.hyperparams['dec_rnn_dim']+self.hyperparams['enc_grasp_dim'], 1))
            nn.init.constant_(self.node_modules['/attractor'].bias, -5.0)  # So that initial sigmoid outputs are small
            

        
        if self.hyperparams['pcl_encoding']:
            self.add_submodule('/pcl_encoder', 
                               model=load_miniptv3(self.hyperparams['enc_pcl_dim']))
            self.add_submodule('/pcl_decoder',
                               model=PointCloudTransformerDecoder(tgt_dim=self.hyperparams['enc_rnn_dim_history'], model_dim=self.hyperparams['enc_pcl_dim'], num_layers=1, output_dim=self.hyperparams['enc_pcl_dim']))
            self.pcl_output_dims = self.hyperparams['enc_pcl_dim']


        ###########################
        #   Node Future Encoder   #
        ###########################
        # We'll create this here, but then later check if in training mode.
        # Based on that, we'll factor this into the computation graph (or not).
        self.add_submodule('/future_encoder',
                           model=nn.LSTM(input_size=self.pred_state_length,
                                                   hidden_size=self.hyperparams['enc_rnn_dim_future'],
                                                   bidirectional=True,
                                                   batch_first=True))
        # These are related to how you initialize states for the node future encoder.
        self.add_submodule('/future_encoder/initial_h',
                           model=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))
        self.add_submodule('/future_encoder/initial_c',
                           model=nn.Linear(self.state_length,
                                                     self.hyperparams['enc_rnn_dim_future']))


        ################################
        #   Discrete Latent Variable   #
        ################################
        self.latent = DiscreteLatent(self.hyperparams, self.device)

        ######################################################################
        #   Various Fully-Connected Layers from Encoder to Latent Variable   #
        ######################################################################
        # Node History Encoder
        x_size = self.hyperparams['enc_rnn_dim_history']
        
        if self.hyperparams['grasp_encoding']:
            x_size += self.grasp_output_dims
        if self.hyperparams['pcl_encoding']:
            x_size += self.pcl_output_dims
        if self.hyperparams['js_encoding']:
            x_size += self.hyperparams['enc_rnn_dim_history']

        z_size = self.hyperparams['N'] * self.hyperparams['K']

        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            self.add_submodule('/p_z_x',
                               model=nn.Linear(x_size, self.hyperparams['p_z_x_MLP_dims']))
            hx_size = self.hyperparams['p_z_x_MLP_dims']
        else:
            hx_size = x_size

        self.add_submodule('/hx_to_z',
                           model=nn.Linear(hx_size, self.latent.z_dim))

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            self.add_submodule('/q_z_xy',
                               # Node Future Encoder
                               model=nn.Linear(x_size + 4 * self.hyperparams['enc_rnn_dim_future'],
                                                         self.hyperparams['q_z_xy_MLP_dims']))
            hxy_size = self.hyperparams['q_z_xy_MLP_dims']
        else:
            #                           Node Future Encoder
            hxy_size = x_size + 4 * self.hyperparams['enc_rnn_dim_future']

        self.add_submodule('/hxy_to_z',
                           model=nn.Linear(hxy_size, self.latent.z_dim))

        ####################
        #   Decoder LSTM   #
        ####################
        decoder_input_dims = self.pred_state_length + z_size + x_size

        self.add_submodule('/decoder/state_action',
                           model=nn.Sequential(
                               nn.Linear(self.state_length, self.pred_state_length)))

        self.add_submodule( '/decoder/rnn_cell',
                           model=nn.GRUCell(decoder_input_dims, self.hyperparams['dec_rnn_dim']))
        self.add_submodule('/decoder/initial_h',
                           model=nn.Linear(z_size + x_size, self.hyperparams['dec_rnn_dim']))

        ###################
        #   Decoder GMM   #
        ###################
        self.add_submodule('/decoder/proj_to_GMM_log_pis',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components']))
        self.add_submodule('/decoder/proj_to_GMM_pos_mus',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length//2))
        self.add_submodule('/decoder/proj_to_GMM_pos_log_sigmas',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * self.pred_state_length//2))
        self.add_submodule('/decoder/proj_to_GMM_pos_corrs',
                           model=nn.Linear(self.hyperparams['dec_rnn_dim'],
                                                     self.hyperparams['GMM_components'] * (self.pred_state_length//2+1)))
        

        self.add_submodule('/decoder/proj_to_GMM_rot_mus',
                           model=nn.Linear(decoder_input_dims-self.pred_state_length//2,
                                                     self.hyperparams['GMM_components'] * self.pred_state_length//2))
        self.add_submodule('/decoder/proj_to_GMM_rot_log_sigmas',
                           model=nn.Linear(decoder_input_dims-self.pred_state_length//2,
                                                     self.hyperparams['GMM_components'] * self.pred_state_length//2))
        self.add_submodule('/decoder/proj_to_GMM_rot_corrs',
                           model=nn.Linear(decoder_input_dims-self.pred_state_length//2,
                                                     self.hyperparams['GMM_components'] * (self.pred_state_length//2+1)))


        self.x_size = x_size
        self.z_size = z_size
        


    def create_new_scheduler(self, name, annealer, annealer_kws, creation_condition=True):
        value_scheduler = None
        rsetattr(self, name + '_scheduler', value_scheduler)
        if creation_condition:
            annealer_kws['device'] = self.device
            value_annealer = annealer(annealer_kws)
            rsetattr(self, name + '_annealer', value_annealer)

            # This is the value that we'll update on each call of
            # step_annealers().
            rsetattr(self, name, value_annealer(0).clone().detach())
            dummy_optimizer = optim.Optimizer([rgetattr(self, name)], {'lr': value_annealer(0).clone().detach()})
            rsetattr(self, name + '_optimizer', dummy_optimizer)

            value_scheduler = CustomLR(dummy_optimizer,
                                       value_annealer)
            rsetattr(self, name + '_scheduler', value_scheduler)

        self.schedulers.append(value_scheduler)
        self.annealed_vars.append(name)

    
    def create_graphical_model(self):
        """
        Creates or queries all trainable components.

        :param edge_types: List containing strings for all possible edge types for the node type.
        :return: None
        """
        self.clear_submodules()

        ############################
        #   Everything but Edges   #
        ############################
        self.create_node_models()

        for name, module in self.node_modules.items():
            module.to(self.device)

    def set_annealing_params(self):
        self.schedulers = list()
        self.annealed_vars = list()

        self.create_new_scheduler(name='kl_weight',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['kl_weight_start'],
                                      'finish': self.hyperparams['kl_weight'],
                                      'center_step': self.hyperparams['kl_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['kl_crossover'] / self.hyperparams[
                                          'kl_sigmoid_divisor']
                                  })

        self.create_new_scheduler(name='latent.temp',
                                  annealer=exp_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['tau_init'],
                                      'finish': self.hyperparams['tau_final'],
                                      'rate': self.hyperparams['tau_decay_rate']
                                  })

        self.create_new_scheduler(name='latent.z_logit_clip',
                                  annealer=sigmoid_anneal,
                                  annealer_kws={
                                      'start': self.hyperparams['z_logit_clip_start'],
                                      'finish': self.hyperparams['z_logit_clip_final'],
                                      'center_step': self.hyperparams['z_logit_clip_crossover'],
                                      'steps_lo_to_hi': self.hyperparams['z_logit_clip_crossover'] / self.hyperparams[
                                          'z_logit_clip_divisor']
                                  },
                                  creation_condition=self.hyperparams['use_z_logit_clipping'])
        self.create_new_scheduler(
            name="gamma",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": self.hyperparams["gamma_init"],
                "finish": self.hyperparams["gamma_end"],
                "center_step": self.hyperparams["gamma_crossover"],
                "steps_lo_to_hi": self.hyperparams["gamma_crossover"]
                / self.hyperparams["gamma_sigmoid_divisor"],
            },
        )

    def step_annealers(self):
        # This should manage all of the step-wise changed
        # parameters automatically.
        for idx, annealed_var in enumerate(self.annealed_vars):
            if rgetattr(self, annealed_var + '_scheduler') is not None:
                # First we step the scheduler.
                with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                    warnings.simplefilter("ignore")
                    rgetattr(self, annealed_var + '_scheduler').step()

                # Then we set the annealed vars' value.
                rsetattr(self, annealed_var, rgetattr(self, annealed_var + '_optimizer').param_groups[0]['lr'])

    
    def obtain_encoded_tensors(self,
                               mode,
                               past_poses,
                               past_joints,
                               future_poses,
                               first_history_indices,
                               context) -> torch.Tensor:

        x, y_e, y = None, None, None
        initial_dynamics = dict()

        batch_size = past_poses.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history = past_poses
        node_present_state = past_poses[:, -1]
        node_pos = past_poses[:, -1, 0:self.pred_state_length]
        node_vel = past_poses[:, -1, self.pred_state_length:2*self.pred_state_length]

        # n_s_t0 = node_present_state_st

        initial_dynamics['pos'] = node_pos
        initial_dynamics['vel'] = node_vel

        self.dynamic.set_initial_condition(initial_dynamics)


        ##################
        # Encode History #
        ##################
        history_encoded = self.encode_node_history(mode,
                                                    node_history,
                                                    first_history_indices)
        

        if self.hyperparams['js_encoding']:
            js_history_encoded = self.encode_js_history(mode,
                                                        past_joints[...,:7],
                                                        first_history_indices)
            # history_encoded = torch.cat((history_encoded, js_history_encoded), dim=-1)
        


        ##################
        # Encode Present #
        ##################
        node_present = n_s_t0 = node_present_state  # [bs, state_dim]

        ##################
        # Encode Future #
        ##################
        if mode != ModeKeys.PREDICT:
            y = future_poses
        

        ##############################
        # Encode Node Edges per Type #
        ##############################
        if self.hyperparams['grasp_encoding'] or self.hyperparams['pcl_encoding']:
            encoded_context = self.encode_context(history_encoded, context)
        
        t4 = time.time()

        ######################################
        # Concatenate Encoder Outputs into x #
        ######################################
        x_concat_list = list()

        if self.hyperparams['js_encoding']:
            x_concat_list.append(js_history_encoded)
        
        if self.hyperparams['grasp_encoding'] or self.hyperparams['pcl_encoding']:
            x_concat_list.append(encoded_context)

        # Every node has a history encoder.
        x_concat_list.append(history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        x = torch.cat(x_concat_list, dim=1)

        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y_e = self.encode_node_future(mode, node_present, y[...,6:])
        
        t5 = time.time()
        
        if torch.isnan(x).sum():
            print()
        
        # print("node history:", t2-t1)
        # print("js history:", t3-t2)
        # print("context:", t4-t3)
        # print("future:", t5-t4)

        return x, y_e, y, n_s_t0
    
    
    def encode_context(self, history_encoded, context):
        """
        Encodes the context information.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param context: Context information for the node.
        :return: Encoded context tensor.
        """
        t1 = time.time()
        context_feature = []
        if self.hyperparams["grasp_encoding"]:
            grasps, mask = padding(context['grasp'])
            context['padded_grasp'] = grasps
            context['padded_grasp_mask'] = mask
            t2 = time.time()
            if grasps.shape[1] == 0:
                grasps = torch.zeros((grasps.shape[0], 1, grasps.shape[2]), device=grasps.device)
                mask = torch.zeros((mask.shape[0], 1), device=mask.device).bool()
            dec_grasp_feature = self.node_modules['/grasp_encoder'](grasps[...,:3], grasps[...,3:], mask, history_encoded, return_intermediate=False)
            context_feature.append(dec_grasp_feature)
            # context['enc_grasp_features'] = enc_grasp_features
        t3 = time.time()
        if self.hyperparams['pcl_encoding']:
            data = self.prepare_ptv3_batch(context['pcl'])
            t4 = time.time()
            out = self.node_modules['/pcl_encoder'](data)
            context['coord'] = out['coord']
            context['batch'] = out['batch']
            pcl_feature, pcl_feat_mask = ptv3_pad_features(out['feat'], out['batch'])
            t5 = time.time()
            pcl_feature = self.node_modules['/pcl_decoder'](history_encoded, pcl_feature, pcl_feat_mask)
            t6 = time.time()
            # pcl_feature = self.pooling_ptv3_output(out)
            context_feature.append(pcl_feature)
        # print("grasp padding:", t2-t1)
        # print("grasp encoding:", t3-t2)
        # print("prepare pcl:", t4-t3)
        # print("pad pcl:", t5-t4)
        # print("decoder:", t6-t5)
        
        context_feature = torch.cat((context_feature), dim=-1)
        return context_feature
    
    def prepare_ptv3_batch(self, pcl_list):
        batches = []
        device = pcl_list[0].device
        for i in range(len(pcl_list)):
            # if pcl_list[i].shape[0] < 50:
            #     continue
            batch_id = i * torch.ones(pcl_list[i].shape[0], dtype=torch.long).to(device)
            batches.append(batch_id)
        pcl = torch.cat(pcl_list, dim=0)
        batches = torch.cat(batches, dim=0)
        data_dict = {
            'feat': pcl,
            'coord': pcl,
            'grid_size': 0.3/40,
            'batch': batches,
        }
        return data_dict
    
    def pooling_ptv3_output(self, output):
        features = output['feat']
        batch_idx = output['batch']
        B = batch_idx.max().item() + 1
        C = features.size(1)
        sum_features = torch.zeros(B, C, device=features.device)
        sum_features.index_add_(0, batch_idx, features)
        
        counts = torch.bincount(batch_idx, minlength=B).unsqueeze(1)
        
        mean_features = sum_features / counts.clamp(min=1)  # 避免除零
        
        return mean_features
        


    def encode_node_history(self, mode, node_hist, first_history_indices):
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules['/history_encoder'],
                                                    original_seqs=node_hist,
                                                    lower_indices=first_history_indices)

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)

        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_js_history(self, mode, node_hist, first_history_indices):
        outputs, _ = run_lstm_on_variable_length_seqs(self.node_modules['/js_encoder'],
                                                    original_seqs=node_hist,
                                                    lower_indices=first_history_indices)

        outputs = F.dropout(outputs,
                            p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                            training=(mode == ModeKeys.TRAIN))  # [bs, max_time, enc_rnn_dim]

        last_index_per_sequence = -(first_history_indices + 1)

        return outputs[torch.arange(first_history_indices.shape[0]), last_index_per_sequence]

    def encode_node_future(self, mode, node_present, node_future) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        initial_h_model = self.node_modules['/future_encoder/initial_h']
        initial_c_model = self.node_modules['/future_encoder/initial_c']

        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = initial_h_model(node_present)
        initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=self.device)], dim=0)

        initial_c = initial_c_model(node_present)
        initial_c = torch.stack([initial_c, torch.zeros_like(initial_c, device=self.device)], dim=0)

        initial_state = (initial_h, initial_c)

        _, state = self.node_modules['/future_encoder'](node_future, initial_state)
        state = unpack_RNN_state(state)
        state = F.dropout(state,
                          p=1. - self.hyperparams['rnn_kwargs']['dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        return state


    def q_z_xy(self, mode, x, y_e) -> torch.Tensor:
        r"""
        .. math:: q_\phi(z \mid \mathbf{x}_i, \mathbf{y}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :return: Latent distribution of the CVAE.
        """
        xy = torch.cat([x, y_e], dim=1)

        if self.hyperparams['q_z_xy_MLP_dims'] is not None:
            dense = self.node_modules['/q_z_xy']
            h = F.dropout(F.relu(dense(xy)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = xy

        to_latent = self.node_modules['/hxy_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def p_z_x(self, mode, x):
        r"""
        .. math:: p_\theta(z \mid \mathbf{x}_i)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :return: Latent distribution of the CVAE.
        """
        if self.hyperparams['p_z_x_MLP_dims'] is not None:
            dense = self.node_modules['/p_z_x']
            h = F.dropout(F.relu(dense(x)),
                          p=1. - self.hyperparams['MLP_dropout_keep_prob'],
                          training=(mode == ModeKeys.TRAIN))

        else:
            h = x

        to_latent = self.node_modules['/hx_to_z']
        return self.latent.dist_from_h(to_latent(h), mode)

    def project_to_GMM_pos_params(self, tensor) -> torch.Tensor:
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        log_pis = self.node_modules['/decoder/proj_to_GMM_log_pis'](tensor)
        mus = self.node_modules['/decoder/proj_to_GMM_pos_mus'](tensor)
        log_sigmas = self.node_modules['/decoder/proj_to_GMM_pos_log_sigmas'](tensor)
        corrs = torch.tanh(self.node_modules['/decoder/proj_to_GMM_pos_corrs'](tensor))
        # corrs = self.node_modules['/decoder/proj_to_GMM_pos_corrs'](tensor)
        return log_pis, mus, log_sigmas, corrs
    
    def project_to_GMM_rot_params(self, tensor) -> torch.Tensor:
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        # log_pis = self.node_modules['/decoder/proj_to_GMM_rot_log_pis'](tensor)
        mus = self.node_modules['/decoder/proj_to_GMM_rot_mus'](tensor)
        log_sigmas = self.node_modules['/decoder/proj_to_GMM_rot_log_sigmas'](tensor)
        corrs = torch.tanh(self.node_modules['/decoder/proj_to_GMM_rot_corrs'](tensor))
        # corrs = self.node_modules['/decoder/proj_to_GMM_rot_corrs'](tensor)
        return mus, log_sigmas, corrs

    def p_y_xz(self, mode, x, n_s_t0, z_stacked, prediction_horizon,
               num_samples, num_components=1, gmm_mode=False, measure=None, context=None):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y: Future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM3D. If mode is Predict, also samples from the GMM.
        """
        self.state_cache = []
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        cell = self.node_modules['/decoder/rnn_cell']
        initial_h_model = self.node_modules['/decoder/initial_h']

        initial_state = initial_h_model(zx)

        log_pis, pos_mus, pos_covs, rot_mus, rot_covs  = [], [], [], [], []

        # Infer initial action state for node from current state
        a_0 = self.node_modules['/decoder/state_action'](n_s_t0)
        self.a_0 =  a_0.repeat(num_samples * num_components, 1)
        state = initial_state

        input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)
        outputs = []
        self.poses_cache = []
        cur_pos = n_s_t0[:, :3].clone()
        cur_pose = n_s_t0[:, :6].clone()
        cur_pose = cur_pose.repeat(num_samples * num_components, 1)
        self.state_cache.append(state)
        
        # if 'enc_grasp_features' in context and 'padded_grasp' in context:
        #     grasp_mask = context['padded_grasp_mask'].repeat(num_samples * num_components, 1)
        #     grasp = context['padded_grasp'].repeat(num_samples * num_components, 1, 1)
        #     context['enc_grasp_features'] = context['enc_grasp_features'].repeat(num_samples * num_components, 1, 1)
        
        for j in range(ph):
            t1 = time.time()
            h_state = cell(input_, state)
            log_pi_t, pos_mu_t, pos_log_sigma_t, pos_corr_t = self.project_to_GMM_pos_params(h_state)

                
            if mode == ModeKeys.PREDICT and context is not None and len(context['pcl']) == 1 and False:
                pseudo_cur_pos = pos_mu_t/self.scale * self.dynamic.dt + cur_pos
                steering_force = self.calculate_repulsive_force(pseudo_cur_pos, context['pcl'][0], eta=0.01, R=0.04)
                # if steering_force.abs().sum() > 0:
                #     print('steering force:', steering_force)
                pos_mu_t += steering_force * self.scale * self.dynamic.dt
                cur_pos = pos_mu_t/self.scale * self.dynamic.dt + cur_pos

            t2 = time.time()

            self.state_cache.append(h_state)
            
            gmm = GMM3D(log_pi_t, pos_mu_t, pos_log_sigma_t, pos_corr_t)  # [k;bs, pred_dim]
            
            try:
                if j == 0 and mode == ModeKeys.PREDICT:
                    self.trans_entropy_lb = self.cal_entropy_lb(gmm, self.latent.p_dist.logits)
                    self.trans_entropy_ub = self.cal_entropy_ub(gmm, self.latent.p_dist.logits)
            except:
                pass
                
            # positional kalman
            if j == 0 and measure is not None:
                if self.ood_steps_after <= 0:
                    in_distribution = True
                else:
                    in_distribution = False
                
                self.ood_steps_after -= 1
                #### ood detection ####
                v_u = measure.mean[0,:3] * self.scale

                Z = gmm.statistical_test(v_u)
                self.ood_buffer.append(Z)
                if min(self.ood_buffer[-self.ood_check_steps:]) > chi2.ppf(self.ood_alpha, df=3):
                    in_distribution = False
                    self.ood_steps_after = 3
                if in_distribution:
                    pred_cov = gmm.cov.reshape(num_components, self.pred_state_length//2, self.pred_state_length//2)
                    user_cov = measure.covariance_matrix[...,:3,:3].clone()*self.scale**2
                    K = torch.bmm(pred_cov, torch.linalg.inv(pred_cov + user_cov))
                    new_mus = gmm.mus.reshape(num_components, self.pred_state_length//2, 1) + torch.bmm(K, (measure.mean[...,:3] * self.scale - gmm.mus.reshape(num_components, self.pred_state_length//2)).reshape(num_components, self.pred_state_length//2, 1))
                    new_cov = pred_cov - torch.bmm(K, pred_cov)
                    new_mus = new_mus.reshape(num_components, self.pred_state_length//2)
                    
                    
                    pis = self.latent.p_dist.logits.exp().reshape(-1)
                    v_mode = gmm.mus[:,0,:]
                    p_r_xcvu = td.MultivariateNormal(v_mode.reshape(-1, self.pred_state_length//2), gmm.cov[:,0]+user_cov, validate_args=False)
                    pis_pos_lh = p_r_xcvu.log_prob(v_u.reshape(1,-1).repeat(num_components,1)).exp()

                    # pis_posterior = pis*(pis_lh + 1e-5)
                    # pis_posterior = pis_posterior/pis_posterior.sum()

                    # self.latent.p_dist = td.Categorical(logits=pis_posterior.log())
                    
                    gmm = gmm.from_log_pis_mus_cov_mats(gmm.log_pis, new_mus, new_cov) #  kf=True
                    pos_mu_t = gmm.mus.squeeze(0)
                else:
                    # print('ood!')
                    gmm = gmm.from_log_pis_mus_cov_mats(gmm.log_pis, measure.mean[:,:3].repeat(num_components,1), measure.covariance_matrix[:,:3,:3].repeat(num_components,1,1)) #  kf=True
                    pos_mu_t = gmm.mus.squeeze(0)

            t3 = time.time()

            if mode == ModeKeys.PREDICT and gmm_mode:
                pos_a_t = gmm.mode()
            else:
                pos_a_t = gmm.rsample()

            
            pos_mus.append(
                pos_mu_t.reshape(
                    num_samples, num_components, -1, (self.pred_state_length//2)
                ).permute(0, 2, 1, 3).reshape(-1, (self.pred_state_length//2) * num_components)
            )
            pos_covs.append(
                gmm.cov.reshape(
                    num_samples, num_components, -1, (self.pred_state_length//2), (self.pred_state_length//2)
                ).permute(0, 2, 1, 3, 4).reshape(-1, num_components * (self.pred_state_length//2) * (self.pred_state_length//2)))

            # pos_corrs.append(
            #     pos_corr_t.reshape(
            #         num_samples, num_components, -1
            #     ).permute(0, 2, 1).reshape(-1, (self.pred_state_length//2+1)* num_components))


            # dec_inputs = [zx, mu_t]
            rot_inputs = [zx, pos_a_t]
            rot_inputs = torch.cat(rot_inputs, dim=1)
            rot_mu_t, rot_log_sigma_t, rot_corr_t = self.project_to_GMM_rot_params(rot_inputs)
            rot_gmm = GMM3D(log_pi_t, rot_mu_t, rot_log_sigma_t, rot_corr_t)

            t4 = time.time()
            try:
                if j == 0 and mode == ModeKeys.PREDICT:
                    self.rot_entropy_lb = self.cal_entropy_lb(rot_gmm, self.latent.p_dist.logits)
                    self.rot_entropy_ub = self.cal_entropy_ub(rot_gmm, self.latent.p_dist.logits)
            except:
                pass
            
            # rotation kalman
            # if in_distribution:
            if j == 0 and measure is not None: #  and torch.norm(measure.mean[...,3:])>1e-3
                w_u = measure.mean[0,3:]
                pred_rot_cov = rot_gmm.cov.reshape(num_components, self.pred_state_length//2, self.pred_state_length//2)
                user_cov = measure.covariance_matrix[...,3:,3:]
                K = torch.bmm(pred_rot_cov, torch.linalg.inv(pred_rot_cov + user_cov))
                new_rot_mus = rot_gmm.mus.reshape(num_components, self.pred_state_length//2, 1) + torch.bmm(K, (measure.mean[...,3:] - rot_gmm.mus.reshape(num_components, self.pred_state_length//2)).reshape(num_components, self.pred_state_length//2, 1))
                new_rot_cov = pred_rot_cov - torch.bmm(K, pred_rot_cov)
                new_rot_mus = new_rot_mus.reshape(num_components, self.pred_state_length//2)

                rot_gmm = rot_gmm.from_log_pis_mus_cov_mats(gmm.log_pis, new_rot_mus, new_rot_cov)
                rot_mu_t = rot_gmm.mus.squeeze(0)
                
                if in_distribution:
                    w_mode = rot_gmm.mus[:,0,:]
                    p_r_xcwu = td.MultivariateNormal(w_mode.reshape(-1, self.pred_state_length//2), rot_gmm.cov+user_cov, validate_args=False)
                    pis_rot_lh = p_r_xcwu.log_prob(w_u.reshape(1,-1).repeat(num_components,1)).exp()
                
                    pis_posterior = pis*(pis_pos_lh + 1e-5)*(pis_rot_lh + 1e-5)
                    pis_posterior = pis_posterior/pis_posterior.sum()
                    self.latent.p_dist = td.Categorical(logits=pis_posterior.log())
                
            # else:
            #     rot_gmm = rot_gmm.from_log_pis_mus_cov_mats(gmm.log_pis, measure.mean[:,3:].repeat(num_components,1), measure.covariance_matrix[:,3:,3:].repeat(num_components,1,1)) #  kf=True
            #     rot_mu_t = rot_gmm.mus.squeeze(0)
            

            if mode == ModeKeys.PREDICT and gmm_mode:
                rot_a_t = rot_gmm.mode()
            else:
                rot_a_t = rot_gmm.rsample()

            
            rot_mus.append(
                rot_mu_t.reshape(
                    num_samples, num_components, -1, (self.pred_state_length//2)
                ).permute(0, 2, 1, 3).reshape(-1, num_components * (self.pred_state_length//2))
            )
            rot_covs.append(
                rot_gmm.cov.reshape(
                    num_samples, num_components, -1, (self.pred_state_length//2), (self.pred_state_length//2)
                ).permute(0, 2, 1, 3, 4).reshape(-1, num_components * (self.pred_state_length//2) * (self.pred_state_length//2)))

            # cur_action = torch.cat((pos_a_t/self.scale, rot_a_t), dim=-1)
            # if 'enc_grasp_features' in context and 'padded_grasp' in context:
            #     grasp_mask = context['padded_grasp_mask'].repeat(num_samples * num_components, 1)
            #     grasp = context['padded_grasp'].repeat(num_samples * num_components, 1, 1)
            #     # context['enc_grasp_features'] = context['enc_grasp_features'].repeat(num_samples * num_components, 1, 1)
            #     cur_pose_term = self.dynamic.integrate_one_step(cur_action, cur_pose)
            #     attractor_input = torch.cat([h_state.unsqueeze(1).repeat(1, context['enc_grasp_features'].shape[1], 1), context['enc_grasp_features'].repeat(num_samples * num_components, 1, 1)], dim=-1)
            #     # attractor_coeff = torch.zeros_like(grasp_mask).float().unsqueeze(-1)
            #     attractor_coeff = self.node_modules['/attractor'](attractor_input).sigmoid()
            #     # attractor_coeff = self.node_modules['/attractor'](context['enc_grasp_features'], h_state.unsqueeze(1).repeat(1, context['enc_grasp_features'].shape[1], 1), grasp_mask).sigmoid()
            #     attractive_twist = relative_pose_logmap_se3(grasp, cur_pose_term) # (bs*n_c*n_s, n_grasp, 6)
            #     attractive_twist = torch.where(attractive_twist.isnan(), torch.zeros_like(attractive_twist), attractive_twist)
            #     attractive_twist = (attractor_coeff * attractive_twist * grasp_mask.unsqueeze(-1).float()).sum(dim=1) # (bs*n_c*n_s, 6)
            #     cur_action += attractive_twist
            # cur_pose = self.dynamic.integrate_one_step(cur_action, cur_pose)
            
            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(pos_log_sigma_t[...,0].reshape(num_samples, num_components, -1).permute(0, 2, 1).reshape(-1, 1))
                )
            
            outputs.append(torch.cat((pos_a_t, rot_a_t), dim=-1))

            dec_inputs = [zx, pos_a_t, rot_a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state
            
            

            # print("lstm cell+trans gmm params:", t2-t1)
            # print("kalman filter trans:", t3-t2)
            # print("rot gmm params:", t4-t3)
            # print("kalman filter rot:", t5-t4)
            # print("one step:", t5-t1)


        log_pis = torch.stack(log_pis, dim=1)
        pos_mus = torch.stack(pos_mus, dim=1)
        rot_mus = torch.stack(rot_mus, dim=1)
        pos_covs = torch.stack(pos_covs, dim=1)
        rot_covs = torch.stack(rot_covs, dim=1)
        outputs = torch.stack(outputs, dim=1) # (n_samples*n_components*batch_size, ph, dim)

        pos_mus = pos_mus.reshape(num_samples, -1, ph, num_components, pred_dim//2)
        rot_mus = rot_mus.reshape(num_samples, -1, ph, num_components, pred_dim//2)
        pos_covs = pos_covs.reshape(num_samples, -1, ph, num_components, (pred_dim//2), (pred_dim//2))
        if mode == ModeKeys.PREDICT:
            pos_mus = pos_mus / self.scale
            pos_covs = pos_covs / (self.scale**2)
        mus = torch.cat((pos_mus, rot_mus), dim=-1)
        rot_covs = rot_covs.reshape(num_samples, -1, ph, num_components, (pred_dim//2), (pred_dim//2))
        covs  = torch.zeros((num_samples, mus.shape[1], ph, num_components, pred_dim, pred_dim), device=mus.device)
        covs[...,:pred_dim//2,:pred_dim//2] = pos_covs
        covs[...,pred_dim//2:,pred_dim//2:] = rot_covs            
        a_dist = GMM6D.from_log_pis_mus_cov_mats(torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
                    torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
                    torch.reshape(covs, [num_samples, -1, ph, num_components, pred_dim, pred_dim]),)

        if self.hyperparams['dynamic']['distribution']:
            y_dist = self.dynamic.integrate_distribution(a_dist, n_s_t0)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                # if len(context['pcl']) == 1:
                #     a_sample = a_dist.mode_colfree(cur_pos, context['pcl'][0])
                # else:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            sampled_future = self.dynamic.integrate_samples(a_sample, n_s_t0)
            return y_dist, a_dist, sampled_future
        else:
            return y_dist, outputs.reshape(num_samples*num_components,-1,ph,pred_dim)
    

    def cal_entropy_lb(self, gmm, log_pi_t):
        cov = gmm.get_covariance_matrix().reshape(-1, 1, gmm.dimensions, gmm.dimensions)
        mus = gmm.mus.reshape(-1, 1, gmm.dimensions)
        n_c, _, dim, _ = cov.shape
        cross_cov = cov[:,None] + cov[None]       # (n_c, n_c, 1, dim, dim)
        cross_mus_diff = (mus[:,None] - mus[None]) # (n_c, n_c, 1, dim)
        cross_cov_inv_flatten = torch.inverse(cross_cov.reshape(-1,dim,dim))
        term = torch.bmm(cross_mus_diff.reshape(-1,1,dim),cross_cov_inv_flatten)
        score = torch.bmm(term, cross_mus_diff.reshape(-1,dim,1)).reshape(n_c, n_c, 1)
        cross_cov_det = torch.det(cross_cov) # (n_c, n_c, 1)
        Z = ((2*torch.pi)**2*cross_cov_det)**(1/2)
        cross_prob = (-0.5*score).exp()*Z.pow(-1) # (n_c, n_c, 1)
        # log_pi_t = log_pi_t - torch.logsumexp(log_pi_t, dim=0, keepdim=True)
        gaussian_alpha = log_pi_t.exp().reshape(-1,1) # (n_c, 1)
        gaussian_alpha = gaussian_alpha / gaussian_alpha.sum()
        entropy = (cross_prob * gaussian_alpha.unsqueeze(-2)).sum(0).log() # (n_c, 1)
        entropy_lb = - (entropy*gaussian_alpha).sum()
        # if entropy_lb < 0:
        #     print()
        return entropy_lb

    def cal_entropy_ub(self, gmm, log_pi_t):
        cov = gmm.get_covariance_matrix().reshape(-1, 1, gmm.dimensions, gmm.dimensions)
        n_c, _, dim, _ = cov.shape
        Z = ((2*torch.pi*torch.e)**2)*torch.det(cov) # (n_c, 1)
        Z = (1/2)*torch.log(Z) - log_pi_t.reshape(-1,1) # (n_c, 1)
        # log_pi_t = log_pi_t - torch.logsumexp(log_pi_t, dim=0, keepdim=True)
        gaussian_alpha = log_pi_t.exp().reshape(-1,1) # (n_c, 1)
        entropy_ub = (Z * gaussian_alpha).sum()
        # if entropy_lb < 0:
        #     print()
        return entropy_ub
    

    def calculate_repulsive_force(
            self,
            agent_pos: torch.Tensor,
            pcl: torch.Tensor,
            R: float = 0.04,
            k: int=15,
            eta: float = 0.1,
            F_max: float = 10.0
        ) -> torch.Tensor:
        """
        计算作用在运动点上的斥力场合力。

        Args:
            agent_pos: 运动点的位置张量。形状: (D,) 或 (M, D)
            pcl: point_clouds 障碍物位置张量。形状: (N, D)，其中 K 是障碍物数量，D 是维度 (2D 或 3D)。
            R: 斥力作用的安全距离/影响半径 (float)。
            eta: 斥力增益系数 (float)。

        Returns:
            torch.Tensor: 作用在运动点上的合力张量。形状: (D,)
        """
        
        # 确保 agent_pos 的形状是 (1, D) 以方便广播运算
        if agent_pos.dim() == 1:
            agent_pos = agent_pos.unsqueeze(0)
        
        # D: 维度 (2D 或 3D), K: 障碍物数量
        # agent_pos 形状: (1, D)
        # obs_pos_k 形状: (K, D)
        dist = torch.cdist(agent_pos, pcl, p=2)  # 计算欧氏距离，形状: (M, N)
        knn_idx = torch.topk(dist, k, largest=False).indices  # 取最近的 k 个障碍物索引，形状: (M, k)
        obs_pos_k = pcl[knn_idx.squeeze(0)] 

        # 1. 计算向量 (P - Ok)
        # shape: (K, D)
        diff = agent_pos[:,None,:] - obs_pos_k  # (M, 1, D) - (M, K, D) -> (M, K, D)

        # 2. 计算距离的平方和距离 dk
        # dist_sq 形状: (K,)
        dist_sq = torch.sum(diff ** 2, dim=-1)
        
        # 防止除以零：在数值上增加一个极小值，虽然实际中障碍物不应与点完全重合
        # dist_sq = dist_sq + 1e-6 
        
        # dist 形状: (K,)
        dist = torch.sqrt(dist_sq) # (M, K)

        # 3. 计算斥力方向单位向量 uk
        # u_k 形状: (K, D)
        # 先 unsqueeze(1) 扩展 dist 的维度到 (K, 1)，使其可以和 diff 进行逐元素除法
        u_k = diff / dist.unsqueeze(-1)
        
        # 4. 计算斥力大小的标量部分 |F_rep, k|
        
        # 创建一个布尔掩码，识别作用范围内的障碍物 (d_k <= R)
        # mask 形状: (M, K)
        mask = (dist <= R) 

        # 根据公式计算斥力大小：eta * (1/d - 1/R) * (1/d^2)
        # 1/d 形状: (K_valid,)
        inv_dist = 1.0 / dist # (M, K)
        inv_dist_sq = inv_dist ** 2
        
        # F_rep_mag_valid 形状: (K_valid,)
        F_rep_mag = eta * (inv_dist - 1.0 / R) * inv_dist_sq
        F_rep_mag = torch.clamp(F_rep_mag, min=0.0, max=F_max)  # 确保斥力大小非负

        # 5. 计算每个障碍物的斥力矢量 F_rep, k
        
        # F_k_valid 形状: (K_valid, D)
        # 扩展 F_rep_mag_valid 的维度到 (K_valid, 1) 进行乘法
        F_k_valid = torch.zeros_like(u_k)
        F_k_valid[mask] = F_rep_mag.unsqueeze(-1)[mask] * u_k[mask]
        
        # 6. 计算合力 F_rep
        # 求和得到最终的合力
        F_rep = torch.mean(F_k_valid, dim=-2)
        return F_rep

    def encoder(self, mode, x, y_e, num_samples=None):
        """
        Encoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :param num_samples: Number of samples from the latent space during Prediction.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams['k']
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams['k_eval']
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        self.latent.q_dist = self.q_z_xy(mode, x, y_e)
        self.latent.p_dist = self.p_z_x(mode, x)

        z = self.latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN:
            kl_obj = self.latent.kl_q_p()
        else:
            kl_obj = None

        return z, kl_obj

    def decoder(self, mode, x, y, n_s_t0, z, labels, prediction_horizon, num_samples, context={}):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y: Future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        y_dist, outputs = self.p_y_xz(mode, x, n_s_t0, z,
                             prediction_horizon, num_samples, num_components=num_components, context=context)
        log_p_yt_xz = torch.clamp(y_dist.log_prob(labels), max=self.hyperparams['log_p_yt_xz_max'])
        # log_p_yt_xz = torch.clamp(y_dist.CVaR_log_prob(labels, alpha=0.8), max=self.hyperparams['log_p_yt_xz_max']) # self.gamma
        
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        return log_p_y_xz, outputs, y_dist

    

    def reward_function(self, samples_trajectories, context):
        """
        trajectories: torch.tensor(batch_size, N_samples, horizon, dim)
        context: {
            obstacles: [torch.tensor(N_obs, dim)],
            goals: [torch.tensor(N_goals, dim)]
        }
        """
        batch_size, N_samples, horizon, _, _ = samples_trajectories.shape
        reward = torch.zeros(batch_size, N_samples).to(self.device)
        terminal_step = torch.ones(batch_size, N_samples).long().to(self.device) * self.ph

        grasps, grasp_mask = padding(context['grasp'])
        batch_pcl_trim = []
        for pcl in context['pcl']:
            neihbor_mask = torch.norm(pcl, dim=-1) < 0.1
            batch_pcl_trim.append(pcl[neihbor_mask])
        obstacles, pcl_mask = padding(batch_pcl_trim, fill_value=float('nan'))

        # grasps = SO3_R3().exp_map(grasps.reshape(-1,6)).to_matrix().reshape(grasps.shape[:-1]+(4,4))
        # grasps = grasps.unsqueeze(1).repeat(1, N_samples, 1, 1, 1) # (batch_size, N_goals, 4, 4)
        
        # pairwise_goals_distance = se3_pairwise_distance(samples_trajectories, grasps)

        # pairwise_goals_distance[torch.isnan(pairwise_goals_distance)] = self.goal_threshold + 1
        # mingoal_distance = pairwise_goals_distance.min(-1)[0] #(batch_size, N_samples, horizon)
        # min_distance, arrived_step = mingoal_distance.min(-1) #(batch_size, N_samples, )
        # arrived_mask = min_distance < self.goal_threshold
        # reward[arrived_mask] = self.goal_reward
        # terminal_step[arrived_mask] = arrived_step[arrived_mask]

        samples_trajectories_pos = samples_trajectories[...,:3,3]
        pairwise_obs_distance = torch.norm(obstacles[:, None, None].to(self.device)-samples_trajectories_pos[:,:, :, None], p=2, dim=-1) #(N_samples, horizon, N_obs)
        pairwise_obs_distance[torch.isnan(pairwise_obs_distance)] = self.col_threshold + 1
        minobs_distance = pairwise_obs_distance.min(-1)[0] 
        min_distance, crash_step = minobs_distance.min(-1) #(N_samples, )
        crash_mask = min_distance < self.col_threshold
        reward[crash_mask] = self.col_reward
        terminal_step[crash_mask] = crash_step[crash_mask]

        #################### viz #######################
        # from dataset.se3_preprocessing import plot_se3_poses
        # fig = plt.figure(figsize=(8, 8))
        # ax = fig.add_subplot(111, projection='3d')
        # for i in range(10):
        #     if (reward[0, i] > 0):
        #         color = 'orange'
        #     elif (reward[0, i] == 0):
        #         color = 'black'
        #     else:
        #         color = 'red'
        #     if i == 0:
        #         plot_se3_poses(samples_trajectories[0,0].cpu().numpy(), SO3_R3().exp_map(context['grasp'][0]).to_matrix().cpu().numpy(), ax=ax, color=color) #pcl=pcl
        #     else:
        #         plot_se3_poses(samples_trajectories[0,i].cpu().numpy(),  ax=ax, color=color) #pcl=pcl
        # plt.show()
        #################### viz #######################

        return reward, terminal_step
    
    
    def GAE(self, Qs, rewards, gamma_, lambda_, terminal_steps):
        """
        Qs: torch.tensor(N_samples*batch_size, horizon+1)
        rewards: torch.tensor(N_samples*batch_size, horizon+1)
        """
        delta = torch.zeros_like(Qs[...,:-1]) # (N_samples, batch_size, horizon)
        delta = rewards[...,:-1] + gamma_*Qs[...,1:] - gamma_*Qs[...,:-1]
        N_steps = rewards.shape[-1]
        advantage = torch.zeros_like(Qs)
        for i in reversed(range(0, advantage.shape[-1]-1)):
            advantage[...,i] = (delta[...,i] + (gamma_*lambda_)*advantage[...,i+1]) * (terminal_steps>=i).float()
        return advantage[...,:N_steps]



    def train_loss(self,
                   past_poses,
                   past_joints,
                   first_history_indices,
                   future_poses,
                   prediction_horizon,
                   context={}
                   ) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param prediction_horizon: Number of prediction timesteps.
        :return: Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN
        future_poses[..., 6:9] = future_poses[..., 6:9] * self.scale
        future_joints = past_joints[:, past_poses.shape[1]:]
        past_joints = past_joints[:, :past_poses.shape[1]]
        x, y_e, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        past_poses=past_poses,
                                                        past_joints=past_joints,
                                                        future_poses=future_poses,
                                                        first_history_indices=first_history_indices,
                                                        context=context)
        
        t1 = time.time()
        

        z, kl = self.encoder(mode, x, y_e)
        log_p_y_xz, samples_actions, y_dist = self.decoder(mode, x, y, n_s_t0, z,
                                  future_poses[...,6:],  # Loss is calculated on unstandardized label
                                  prediction_horizon,
                                  self.hyperparams['k'],
                                  context=context)

        # manipulability = manipulability_panda(future_joints[...,:7].reshape(-1,7)).reshape(future_joints.shape[:-1])


        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        # mutual_inf_q = mutual_inf_mc(self.latent.q_dist, None)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist, None)

        ELBO =  - self.kl_weight * kl + 1. * mutual_inf_p + log_likelihood
        loss = -ELBO

        ### regularization loss ###
        # lambda_reg = 0.1
        # reg_margin = 0.3
        # mus = y_dist.mus
        # cross_mus_diff = torch.norm(mus[:,:,:,None] - mus[:,:,:,:,None], dim=-1).mean(2).mean(0) # (bs, n_c, n_c)
        # cross_mus_diff = torch.clip(cross_mus_diff, min=reg_margin)
        # rbf = torch.exp(- (cross_mus_diff) * lambda_reg)
        # bs, n_c, _ = rbf.shape
        # eye = torch.eye(n_c, device=rbf.device).reshape(1, n_c, n_c).repeat(bs, 1, 1)
        # rbf = rbf * (1 - eye)
        # reg_loss = rbf.mean()
        # loss += reg_loss
        ### regularization loss ###


        ### collision loss ###
        # collision_loss_weight = 0.1
        # n_c, bs, T, dim = samples_actions.shape
        # samples_trajectories = self.dynamic.integrate_samples(samples_actions/self.scale, n_s_t0).permute(1, 0, 2, 3, 4)
        # samples_trajectories = samples_trajectories[..., :3, 3].reshape(bs,-1,3)
        # col_loss = collision_avoidance_regularizer(samples_trajectories, context['coord'], context['batch'], margin=0.05,  margin_loss_weight=collision_loss_weight)
        # loss += col_loss
        ### collision loss ###
        
        
        ### entropy loss ###
        # self.alpha = self.log_alpha.exp()
        # cov = y_dist.get_covariance_matrix()
        # mus = y_dist.mus
        # n_s, bs, ts, n_c, dim, _ = cov.shape
        # cross_cov = cov[:,:,:,None] + cov[:,:,:,:, None] # (1, bs, ts, n_c, n_c, dim, dim)
        # cross_mus_diff = (mus[:,:,:,None] - mus[:,:,:,:,None]) # (1, bs, ts, n_c, n_c, dim)
        # cross_cov_inv_flatten = torch.inverse(cross_cov.reshape(-1,dim,dim))
        # term = torch.bmm(cross_mus_diff.reshape(-1,1,dim),cross_cov_inv_flatten)
        # score = torch.bmm(term, cross_mus_diff.reshape(-1,dim,1)).reshape(n_s, bs, ts, n_c, n_c)
        # cross_cov_det = torch.det(cross_cov) # (1, bs, ts, n_c, n_c)
        # Z = ((2*torch.pi)**2*cross_cov_det)**(1/2)
        # cross_prob = (-0.5*score).exp()*Z.pow(-1)
        # gaussian_alpha = y_dist.log_pis.exp()
        # entropy = (cross_prob * gaussian_alpha.unsqueeze(-2)).sum(-1).log()
        # entropy_lb = - (entropy*gaussian_alpha).sum(-1).mean()
        # loss += self.alpha[0].detach() * (-entropy_lb)
        ### entropy loss ###

        ### RL ###
        RL = False
        if RL == True:
            if self.total_it % self.target_update_interval == 0:
                for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            n_c, bs, T, dim = samples_actions.shape
            samples_trajectories = self.dynamic.integrate_samples(samples_actions/self.scale, n_s_t0).permute(1, 0, 2, 3, 4)

            with torch.no_grad():
                rewards, terminal_steps = self.reward_function(samples_trajectories, context)
                returns = rewards * self.discount_factor**terminal_steps
                returns[terminal_steps>=self.ph] = 0.0
                done_mask = self.done_mask(terminal_steps.reshape(-1),self.ph).bool()
                
            ############# Critic Optimization #########
                
            state_cache = torch.stack(self.state_cache, dim=1)
            samples_actions_ = torch.cat((self.a_0.reshape(-1, 1, self.pred_state_length), samples_actions.reshape(-1, self.ph,  self.pred_state_length)), dim=1).reshape(-1, self.pred_state_length)
            with torch.no_grad():
                target_Q = self.critic_target(state_cache.reshape(-1,self.hyperparams['dec_rnn_dim']), samples_actions_) # (sample_N*batch_size*(ph+1), N_Q)
                reward_t = torch.zeros_like(state_cache[...,0]) # (sample_N*batch_size, ph+1)
                # reward_t = torch.ones_like(state_cache[...,0]) * self.action_reward
                rewards[terminal_steps>=self.ph] = 0.0
                terminal_steps = terminal_steps.permute(1,0).reshape(-1,1).contiguous()
                done_mask = self.done_mask(terminal_steps[...,0],self.ph)
                reward_t.scatter_(1, terminal_steps, rewards.permute(1,0).reshape(-1,1).contiguous()) # (sample_N*batch_size, ph+1)
                target_Q = torch.min(target_Q, -1)[0].reshape(-1, self.ph+1) #  (sample_N*batch_size, ph+1)

                ######### GAE ###################
                advantage = self.GAE(target_Q, reward_t, gamma_=0.95, lambda_=0.98, terminal_steps=terminal_steps[...,0])
                target_Q[:,:-1] = advantage[:,:-1] + target_Q[:,:-1]
                target_Q.scatter_(1, terminal_steps, rewards.permute(1,0).reshape(-1,1).contiguous())
                target_Q = target_Q[:,:-1]
                ######### GAE ###################
                
                #######################
                # target_Q[:,:-1] = reward_t[:,:-1] + self.discount_factor*target_Q[:,1:] # (sample_N*batch_size, ph+1)
                # target_Q.scatter_(1, terminal_steps, rewards.permute(1,0).reshape(-1,1).contiguous())
                # target_Q = target_Q[:,:-1]
                ############################
                target_Q.clip_(-1,1)
            
            Q = self.critic(state_cache.reshape(-1,self.hyperparams['dec_rnn_dim']), samples_actions_) # (sample_N*batch_size*(ph+1), N_Q)
            Q = Q.reshape(-1, self.ph+1, Q.shape[-1])
            
            critic_loss = self.critic_w * (Q[:,:-1] - target_Q.unsqueeze(-1)).pow(2).sum(-1)
            critic_loss = critic_loss[done_mask].mean()
            
            loss += critic_loss
            ############# Critic Optimization #########

            ############# policy optimization #############
            # log_policy = torch.clamp(y_dist.log_prob(samples_actions), max=self.hyperparams['log_p_yt_xz_max']) # (sample_N, batch_size, ph)
            # done_mask = done_mask.reshape(log_policy.permute(1,0,2).shape)
            # actor_loss = - self.pg_alpha*(returns.unsqueeze(-1)*log_policy.permute(1,0,2))[done_mask].mean()
            # loss += actor_loss
            ############# policy optimization #############
            
            
            
            ############# Actor optimization gradient #############           
            # advantage = target_Q - Q[:,:-1].min(-1)[0]
            advantage = advantage[:, :-1]
            ##### normalize advantage ####
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8) # gaussian normalization
            # advantage = ((advantage - advantage.min()) / (advantage.max() - advantage.min() + 1e-8))*2 - 1.0 # gaussian normalization
            ##### normalize advantage ####
            sigma_loss = torch.relu(torch.det(y_dist.cov) - 20).mean()
            loss += sigma_loss
            log_policy = torch.clamp(y_dist.log_prob(samples_actions), max=self.hyperparams['log_p_yt_xz_max']) # (sample_N, batch_size, ph)
            actor_loss = - self.pg_alpha*(advantage.detach()*log_policy.reshape(-1,self.ph))[done_mask].mean()
            # neg_entropy_ref = log_policy.mean()
            
            loss += actor_loss
            
            ############# Actor optimization #############

        return loss


    def done_mask(self, terminal_steps, T):
        done_mask = torch.ones_like(terminal_steps.unsqueeze(-1).repeat(1,T))
        steps = torch.cumsum(torch.ones_like(terminal_steps.unsqueeze(-1).repeat(1,T)),dim=-1) - 1
        done_mask[steps > terminal_steps.unsqueeze(-1).repeat(1,T)] = 0
        return done_mask 


    def eval_loss(self,
                past_poses,
                past_joints,
                first_history_indices,
                future_poses,
                prediction_horizon,
                context={}) -> torch.Tensor:
        """
        Calculates the evaluation loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param prediction_horizon: Number of prediction timesteps.
        :return: tuple(nll_q_is, nll_p, nll_exact, nll_sampled)
        """

        mode = ModeKeys.EVAL
        # future_poses[..., 6:9] = future_poses[..., 6:9] * self.scale
        future_joints = past_joints[:, past_poses.shape[1]:]
        past_joints = past_joints[:, :past_poses.shape[1]]
        x, y_e, y, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        past_poses=past_poses,
                                                        past_joints=past_joints,
                                                        future_poses=future_poses,
                                                        first_history_indices=first_history_indices,
                                                        context=context)

        num_components = self.hyperparams['N'] * self.hyperparams['K']
        ### Importance sampled NLL estimate
        z, _ = self.encoder(mode, x, y_e)  # [k_eval, nbs, N*K]
        z = self.latent.sample_p(1, mode, full_dist=True)
        y_dist, _, _ = self.p_y_xz(ModeKeys.PREDICT, x, n_s_t0, z,
                                prediction_horizon, num_samples=1, num_components=num_components, context=context)
        # We use unstandardized labels to compute the loss
        log_p_yt_xz = torch.clamp(y_dist.log_prob(future_poses[...,6:]), max=self.hyperparams['log_p_yt_xz_max'])
        log_p_y_xz = torch.sum(log_p_yt_xz, dim=2)
        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)
        nll = -log_likelihood

        return nll

    def predict(self,
                past_poses,
                past_joints,
                first_history_indices,
                prediction_horizon,
                num_samples,
                context={},
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                dist=False,
                measure=None):
        """
        Predicts the future of a batch of nodes.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT
        # future_joints = past_joints[:, past_poses.shape[1]:]
        past_joints = past_joints[:, :past_poses.shape[1]]

        x, _, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        past_poses=past_poses,
                                                        past_joints=past_joints,
                                                        future_poses=None,
                                                        first_history_indices=first_history_indices,
                                                        context=context)

        self.latent.p_dist = self.p_z_x(mode, x)
        z, num_samples, num_components = self.latent.sample_p(num_samples,
                                                              mode,
                                                              most_likely_z=z_mode,
                                                              full_dist=full_dist,
                                                              all_z_sep=all_z_sep)

        y_dist, a_dist, our_sampled_future = self.p_y_xz(mode, x, n_s_t0, z,
                                            prediction_horizon,
                                            num_samples,
                                            num_components,
                                            gmm_mode,
                                            measure,
                                            context=context)


        if dist == True:
            return y_dist, a_dist, our_sampled_future
        return our_sampled_future
    
    # def inference_mode(self):
    #     self.node_modules['/grasp_encoder'] = torch.jit.script(self.node_modules['/grasp_encoder'])
        # self.node_modules['/pcl_encoder'] = torch.jit.script(self.node_modules['/pcl_encoder'])
        # self.node_modules['/pcl_decoder'] = torch.jit.script(self.node_modules['/pcl_decoder'])

    
    def get_latent(self,
                   inputs,
                   inputs_st,
                   first_history_indices,
                   labels,
                   labels_st,
                   prediction_horizon,
                   context
                   ) -> torch.Tensor:

        mode = ModeKeys.TRAIN

        x, _, _, n_s_t0 = self.obtain_encoded_tensors(mode=mode,
                                                        inputs=inputs,
                                                        inputs_st=inputs_st,
                                                        labels=labels,
                                                        labels_st=labels_st,
                                                        first_history_indices=first_history_indices,
                                                        context=context)
        return x


