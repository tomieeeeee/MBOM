import torch
import numpy as np
import torch.nn as nn



def get_return(reward, args):
    """Compute episodic return given trajectory

    Args:
        reward (list): Contains rewards across trajectories for specific agent
        args (argparse): Python argparse that contains arguments
    Returns:
        return_ (torch.Tensor): Episodic return with shape: (batch, ep_horizon)
    """
    #reward = torch.stack(reward, dim=1)
    #assert reward.shape == (args.traj_batch_size, args.ep_horizon), \
    #    "Shape must be: (batch, ep_horizon)"

    R, return_ = 0., []
    for timestep in reversed(range(args.ep_horizon)):
        R = reward[:, timestep] + args.discount * R
        return_.insert(0, R)
    return_ = torch.stack(return_, dim=1)

    return return_
class LinearFeatureBaseline(nn.Module):
    """Linear baseline based on handcrafted features, as described in Duan et al., 2016
    Yan Duan, Xi Chen, Rein Houthooft, John Schulman, Pieter Abbeel,
    "Benchmarking Deep Reinforcement Learning for Continuous Control", 2016
    Args:
        input_size (int): Input size from environment
        args (argparse): Python argparse that contains arguments
        reg_coeff (float): Regularization coefficient. Default: 1e-5
    Reference:
        https://github.com/tristandeleu/pytorch-maml-rl/blob/master/maml_rl/baseline.py
    """
    def __init__(self, input_size, args, reg_coeff=1e-5):
        super(LinearFeatureBaseline, self).__init__()

        self.input_size = input_size
        self.args = args
        self._reg_coeff = reg_coeff

        self.weight = nn.Parameter(torch.Tensor(self.feature_size,), requires_grad=False)
        self.weight.data.zero_()
        self._eye = torch.eye(self.feature_size, dtype=torch.float32, device=self.weight.device)

    @property
    def feature_size(self):
        return 2 * self.input_size + 4

    def _feature(self, obs):
        batch_size, sequence_length, _ = obs.shape
        assert sequence_length == self.args.ep_horizon, "Length should equal to episodic horizon"

        ones = torch.ones((sequence_length, batch_size, 1))
        obs = obs.clone().transpose(0, 1)
        time_step = torch.arange(sequence_length).view(-1, 1, 1) * ones / 100.0

        return torch.cat([
            obs,
            obs ** 2,
            time_step,
            time_step ** 2,
            time_step ** 3,
            ones
        ], dim=2)

    def fit(self, obs, return_):
        # featmat.shape: sequence_length * batch_size x feature_size
        featmat = self._feature(obs).view(-1, self.feature_size)

        # returns.shape: sequence_length * batch_size x 1
        returns = return_.view(-1, 1)

        reg_coeff = self._reg_coeff
        XT_y = torch.matmul(featmat.t(), returns)
        XT_X = torch.matmul(featmat.t(), featmat)
        for _ in range(5):
            try:
                coeffs, _ = torch.lstsq(XT_y, XT_X + reg_coeff * self._eye)
                break
            except RuntimeError:
                reg_coeff *= 10
        else:
            raise RuntimeError(
                'Unable to solve the normal equations in '
                '`LinearFeatureBaseline`. The matrix X^T*X (with X the design '
                'matrix) is not full-rank, regardless of the regularization '
                '(maximum regularization: {0}).'.format(reg_coeff))
        self.weight.copy_(coeffs.flatten())

    def forward(self, obs, reward):
        # Fit linear feature baseline
        obs = torch.from_numpy(np.stack(obs, axis=1)).float()
        return_ = get_return(reward, self.args)
        self.fit(obs, return_)

        # Return value
        features = self._feature(obs)
        value = torch.mv(features.view(-1, self.feature_size), self.weight)

        return value.view(features.shape[:2]).transpose(0, 1)

def magic_box(x):
    """DiCE operation that saves computation graph inside tensor
    See ``Implementation of DiCE'' section in the DiCE Paper for details
    Args:
        x (tensor): Input tensor
    Returns:
        1 (tensor): Tensor that has computation graph saved
    References:
        https://github.com/alshedivat/lola/blob/master/lola_dice/rpg.py
        https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
    """
    return torch.exp(x - x.detach())


def get_dice_loss(logprobs, reward, value, args, i_agent, is_train):
    """Compute DiCE loss
    In our code, we use DiCE in the inner loop to be able to keep the dependency in the 
    adapted parameters. This is required in order to compute the opponent shaping term.
    Args:
        logprobs (list): Contains log probability of all agents
        reward (list): Contains rewards across trajectories for specific agent
        value (tensor): Contains value for advantage computed via linear baseline
        args (argparse): Python argparse that contains arguments
        i_agent (int): Agent to compute DiCE loss for
        is_train (bool): Flag to identify whether in meta-train or not
    Returns:
        dice loss (tensor): DiCE loss with baseline reduction
    References:
        https://github.com/alshedivat/lola/blob/master/lola_dice/rpg.py
        https://github.com/alexis-jacq/LOLA_DiCE/blob/master/ipd_DiCE.py
    """
    # Get discounted_reward
    #reward = torch.stack(reward, dim=1)
    cum_discount = torch.cumprod(args.discount * torch.ones(*reward.size()), dim=1) / args.discount
    discounted_reward = reward * cum_discount

    # Compute stochastic nodes involved in reward dependencies
    if args.opponent_shaping and is_train:
        logprob_sum, stochastic_nodes = 0., 0.
        for logprob in logprobs:
            #logprob = torch.stack(logprob, dim=1)
            logprob_sum += logprob
            stochastic_nodes += logprob
        dependencies = torch.cumsum(logprob_sum, dim=1)
    else:
        logprob = logprobs[i_agent]
        #logprob = torch.stack(logprobs[i_agent], dim=1)
        dependencies = torch.cumsum(logprob, dim=1)
        stochastic_nodes = logprob

    # Get DiCE loss
    dice_loss = torch.mean(torch.sum(magic_box(dependencies) * discounted_reward, dim=1))

    # Apply variance_reduction if value is provided
    baseline_term = 0.
    if value is not None:
        discounted_value = value.detach() * cum_discount
        baseline_term = torch.mean(torch.sum((1 - magic_box(stochastic_nodes)) * discounted_value, dim=1))

    return -(dice_loss + baseline_term)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="meta_mapg")

    # Algorithm
    parser.add_argument(
        "--opponent_shaping", type=bool, default=True, #action="store_true",
        help="If True, include opponent shaping in meta_optimization")
    parser.add_argument(
        "--traj_batch_size", type=int, default=150,
        help="Number of trajectories for each inner_loop update (K Hyperparameter)")
    parser.add_argument(
        "--n_process", type=int, default=5,
        help="Number of parallel processes for meta_optimization")
    parser.add_argument(
        "--actor_lr_inner", type=float, default=0.1,
        help="Learning rate for actor (inner loop)")
    parser.add_argument(
        "--actor_lr_outer", type=float, default=0.0001,
        help="Learning rate for actor (outer loop)")
    parser.add_argument(
        "--value_lr", type=float, default=0.00015,
        help="Learning rate for value (outer loop)")
    parser.add_argument(
        "--entropy_weight", type=float, default=0.5,
        help="Entropy weight in the meta-optimization")
    parser.add_argument(
        "--discount", type=float, default=0.96,
        help="Discount factor in reinforcement learning")
    parser.add_argument(
        "--lambda_", type=float, default=0.95,
        help="Lambda factor in GAE computation")
    parser.add_argument(
        "--chain_horizon", type=int, default=5,
        help="Markov chain terminates when chain horizon is reached")
    parser.add_argument(
        "--n_hidden", type=int, default=64,
        help="Number of neurons for hidden network")
    parser.add_argument(
        "--max_grad_clip", type=float, default=10.0,
        help="Max norm gradient clipping value in meta-optimization")

    # Env
    parser.add_argument(
        "--env_name", type=str, default="IPD_v0",
        help="OpenAI gym environment name")
    parser.add_argument(
        "--ep_horizon", type=int, default=30,
        help="Episode is terminated when max timestep is reached")
    parser.add_argument(
        "--n_agent", type=int, default=2,
        help="Number of agents in a shared environment")

    # Misc
    parser.add_argument(
        "--seed", type=int, default=1,
        help="Sets Gym, PyTorch and Numpy seeds")
    parser.add_argument(
        "--test_mode", action="store_true",
        help="If True, perform test during training")
    parser.add_argument(
        "--prefix", type=str, default="",
        help="Prefix for tb_writer and logging")
    args = parser.parse_args()
    num_trj =5
    max_step = 30
    obs = np.random.random((max_step, num_trj, 24))
    logprobs = torch.Tensor(np.random.random((2, num_trj, max_step)))
    reward = torch.Tensor(np.random.random((num_trj, max_step)))

    #obs, logprobs, _, _, rewards = memory.sample()
    linear_baseline = LinearFeatureBaseline(input_size=24, args=args)
    # Compute value for baseline
    value = linear_baseline(obs, reward)  #shape is (num_trj, max_step)
    actor_loss = get_dice_loss(logprobs, reward, value, args, 0, is_train=True)
    a = 1111