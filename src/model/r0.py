import torch

from src.model.r0_base import R0GeneratorBase
from src.model.matrix_generator import generate_transition_matrix, get_infectious_states


class R0Generator(R0GeneratorBase):
    def __init__(self, data, device, n_age: int):
        """
        Initialises R0Generator class instance, for the calculation of the effective reproduction number
        (R0) for the specified model and parameters. It is used for computing the base transmission rate,
        by factoring it out from the NGM (next-generation matrix), and dividing a given R0 by the
        spectral radius (largest eigenvalue) of the NGM.

        Args:
            data:
            device: The device (CPU or GPU) on which the calculations will be performed.
            n_age (int): The number of age groups in the model.

        """
        self.data = data
        super().__init__(data=data, n_age=n_age)

        self.device = device
        self._get_e()

    def _get_v(self) -> None:
        """
        Compute and store the inverse of the transition matrix.

        Returns:
            None
        """
        idx = self._idx
        trans_mtx = generate_transition_matrix(self.inf_state_dict, self.data.trans_data, self.data.model_parameters,
                                               self.n_age, self.n_states, self.i).to(self.device)
        #trans_mtx[idx(f'e_{self.n_e - 1}'), idx('i_0')] = params["alpha"]
        self.v_inv = torch.linalg.inv(trans_mtx)

    def _get_f(self, contact_mtx: torch.Tensor) -> torch.Tensor:
        """
       Compute the matrix representing the rate of infection.

       Args:
           contact_mtx (torch.Tensor): The contact matrix.

       Returns:
           torch.Tensor: The matrix representing the rate of infection.
       """
        i = self.i
        s_mtx = self.s_mtx
        n_states = self.n_states

        infectious_states = get_infectious_states(state_data=self.data.state_data)

        f = torch.zeros((s_mtx, s_mtx)).to(self.device)
        susc_vec = self.parameters["susc"].reshape((-1, 1))
        # Rate of infection for every infectious state
        for inf_state in infectious_states:
            f[i[inf_state]:s_mtx:n_states, i["i_0"]:s_mtx:n_states] = torch.mul(susc_vec, contact_mtx.T)
        return f

    def _get_e(self):
        block = torch.zeros(self.n_states, ).to(self.device)
        block[0] = 1
        self.e = block
        for i in range(1, self.n_age):
            self.e = torch.block_diag(self.e, block)
