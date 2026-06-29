import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import torch
import numpy as np
from model.mgcvae import MultimodalGenerativeCVAE

class Trajectron(object):
    def __init__(self, hyperparams,
                 device, model=None):
        super(Trajectron, self).__init__()
        self.hyperparams = hyperparams
        self.device = device
        self.curr_iter = 0

        self.node_models_dict = dict()
        self.nodes = set()

        self.min_ht = self.hyperparams['minimum_history_length']
        self.max_ht = self.hyperparams['maximum_history_length']
        self.ph = self.hyperparams['prediction_horizon']
        self.state = self.hyperparams['state']
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum([len(entity_dims) for entity_dims in self.state[state_type]])
            )
        self.pred_state = self.hyperparams['pred_state']

        if model is None:
            self.model = MultimodalGenerativeCVAE(self.hyperparams,
                                            self.device)
        else:
            self.model = model
        self.scale = 5
        
        ## calculate total number of parameters
        # total_params = sum(p.numel() for p in self.model.parameters())
        # print(f"Total number of parameters in Trajectron: {total_params}")



    def set_curr_iter(self, curr_iter):
        self.curr_iter = curr_iter
        self.model.set_curr_iter(curr_iter)

    def set_annealing_params(self):
        self.model.set_annealing_params()

    def step_annealers(self):
        self.model.step_annealers()

    def train_loss(self, batch, ph=None):
        if ph is None:
            ph = self.ph
        (first_history_index, x, q, y, context) = batch

        x = x.to(self.device)
        y = y.to(self.device)
        q = q.to(self.device)
        for key in context.keys():
            if isinstance(context[key], list):
                context[key] = [item.to(self.device) for item in context[key]]
            else:
                context[key] = context[key].to(self.device)


        # Run forward pass
        loss = self.model.train_loss(past_poses=x,
                                    past_joints=q,
                                    first_history_indices=first_history_index,
                                    future_poses=y,
                                    prediction_horizon=ph,
                                    context=context)

        return loss

    def eval_loss(self, batch):
        (first_history_index, x, q, y, context) = batch

        x = x.to(self.device)
        y = y.to(self.device)
        q = q.to(self.device)
        for key in context.keys():
            if isinstance(context[key], list):
                context[key] = [item.to(self.device) for item in context[key]]
            else:
                context[key] = context[key].to(self.device)

        # Run forward pass
        # model = self.node_models_dict[node_type]
        nll = self.model.eval_loss(past_poses=x,
                                    past_joints=q,
                                    first_history_indices=first_history_index,
                                    future_poses=y,
                                    prediction_horizon=self.ph,
                                    context=context)

        return nll.cpu().detach().numpy()

    def predict(self,
                batch,
                ph,
                num_samples=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                all_z_sep=False,
                dist=False,
                measure=None):
        (first_history_index, x, q, y, context) = batch

        x = x.to(self.device)
        y = y.to(self.device)
        q = q.to(self.device)
        for key in context.keys():
            if isinstance(context[key], list):
                context[key] = [item.to(self.device) for item in context[key]]
            else:
                context[key] = context[key].to(self.device)

        # Run forward pass
        predictions = self.model.predict(past_poses=x,
                                    past_joints=q,
                                    first_history_indices=first_history_index,
                                    prediction_horizon=ph,
                                    num_samples=num_samples,
                                    context=context,
                                    z_mode=z_mode,
                                    gmm_mode=gmm_mode,
                                    full_dist=full_dist,
                                    all_z_sep=all_z_sep,
                                    dist=dist,
                                    measure=measure)
        if dist == True:
            y_dist, a_dist, predictions = predictions

        predictions_np = predictions.cpu().detach().numpy()

            # Assign predictions to node
            # for i, ts in enumerate(timesteps_o):
            #     if ts not in predictions_dict.keys():
            #         predictions_dict[ts] = dict()
            #     predictions_dict[ts][nodes[i]] = np.transpose(predictions_np[:, [i]], (1, 0, 2, 3))
        if dist:
            return y_dist, a_dist, predictions_np
        return predictions_np




    def get_latent(self, batch):
        (first_history_index,
         x_t, y_t, x_st_t, y_st_t) = batch

        x = x_t.to(self.device)
        y = y_t.to(self.device)
        x_st_t = x_st_t.to(self.device)
        y_st_t = y_st_t.to(self.device)

        # Run forward pass
        # model = self.node_models_dict[node_type]
        feat_x = self.model.get_latent(inputs=x,
                                inputs_st=x_st_t,
                                first_history_indices=first_history_index,
                                labels=y,
                                labels_st=y_st_t,
                                prediction_horizon=self.ph)
        return feat_x


if __name__ == "__main__":
    import os
    import json
    device = torch.cuda.set_device('cuda:0')


    # Load hyperparameters from json
    with open("/home/u0161364/Robot-TrajectronV3/config/config_test.json", 'r', encoding='utf-8') as conf_json:
        hyperparams = json.load(conf_json)

    # Add hyperparams from arguments
    hyperparams['batch_size'] = 128
    hyperparams['k_eval'] = 10
    hyperparams['map_encoding'] = True
    hyperparams['frequency'] = 20
    best_ade = 1000
    dim = 12
    
    
    trajectron = Trajectron(hyperparams, device)
    
    first_history_index = torch.tensor([0,0])
    past_poses = torch.randn((2, 8, dim))  # Example past poses
    past_joints = torch.randn((2, 8, 14))  # Example past
    future_poses = torch.randn((2, 12, dim))  # Example future poses
    context = {'grasp': [torch.randn((10, 9)), torch.randn((12, 9))],
               }# Example context
    trajectron.set_annealing_params()
    trajectron.model.train()
        
    batch = (first_history_index, past_poses, past_joints, future_poses, context)
    trajectron.set_curr_iter(0)
    trajectron.step_annealers()
    train_loss = trajectron.train_loss(batch)

    print()
    
    