import math

import gpytorch
import gpytorch.kernels as kernels
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from gpytorch.constraints import Positive
from gpytorch.distributions import MultivariateNormal
from gpytorch.priors import NormalPrior
from torch.nn.parameter import Parameter

from onescience.utils.GP_TO.plot_latenth import plot_sep

from .gpregression import GPR
from .mollified_uniform import MollifiedUniformPrior


def setlevels(X, qual_index=None, return_label=False):
    labels = []
    if qual_index == []:
        return X
    if qual_index is None:
        qual_index = list(range(X.shape[-1]))
    # if type(X) == np.ndarray:
    #     temp = torch.from_numpy(X).detach().clone()
    temp = np.copy(X)
    if type(X) == torch.Tensor:
        temp = X.clone()
    if temp.ndim > 1:
        for j in qual_index:
            l = np.sort(np.unique(temp[..., j])).tolist()
            labels.append(l)
            # l =  torch.unique(temp[..., j], sorted = True).tolist()
            temp[..., j] = torch.tensor([*map(lambda m: l.index(m), temp[..., j])])
    else:
        l = torch.unique(temp, sorted=True)
        temp = torch.tensor([*map(lambda m: l.tolist().index(m), temp)])

    if temp.dtype == object:
        temp = temp.astype(float)
        if type(X) == np.ndarray:
            temp = torch.from_numpy(temp)

        if return_label:
            return temp, labels
        else:
            return temp
    else:
        if type(X) == np.ndarray:
            temp = torch.from_numpy(temp)
        if return_label:
            return temp, labels
        else:
            return temp


class GPPLUS(GPR):
    """The latent Map GP regression model (LMGP) which extends GPs to handle categorical inputs.

    :note: Binary categorical variables should not be treated as qualitative inputs. There is no
        benefit from applying a latent variable treatment for such variables. Instead, treat them
        as numerical inputs.

    :param train_x: The training inputs (size N x d). Qualitative inputs needed to be encoded as
        integers 0,...,L-1 where L is the number of levels. For best performance, scale the
        numerical variables to the unit hypercube.
    """

    def __init__(
        self,
        # transformation_of_A_parameters:str,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        collocation_x: torch.Tensor,
        qual_ind_lev={},
        multiple_noise=False,
        lv_dim: int = 2,
        quant_correlation_class: str = "Rough_RBF",
        noise: float = 5e-8,
        fix_noise: bool = True,
        lb_noise: float = 1e-8,
        NN_layers: list = [],
        name_output: str = "u",
        encoding_type="one-hot",
        uniform_encoding_columns=2,
        lv_columns=[],
        basis="neural_network",
        NN_layers_base=[4, 4],
        basis_function_size=None,
        device="cpu",
        dtype=torch.float32,
    ) -> None:

        tkwargs = {}  # or dict()
        tkwargs["dtype"] = dtype
        tkwargs["device"] = device

        qual_index = list(qual_ind_lev.keys())
        all_index = set(range(train_x.shape[-1]))
        quant_index = list(all_index.difference(qual_index))
        num_levels_per_var = list(qual_ind_lev.values())
        # ------------------- lm columns --------------------------
        lm_columns = list(set(qual_index).difference(lv_columns))
        if len(lm_columns) > 0:
            qual_kernel_columns = [*lv_columns, lm_columns]
        else:
            qual_kernel_columns = lv_columns

        #########################
        if len(qual_index) > 0:
            train_x = setlevels(train_x, qual_index=qual_index)
        #
        if multiple_noise:
            noise_indices = list(range(0, num_levels_per_var[0]))
        else:
            noise_indices = []

        if len(qual_index) == 1 and num_levels_per_var[0] < 2:
            temp = quant_index.copy()
            temp.append(qual_index[0])
            quant_index = temp.copy()
            qual_index = []
            lv_dim = 0
        elif len(qual_index) == 0:
            lv_dim = 0

        quant_correlation_class_name = quant_correlation_class

        if len(qual_index) == 0:
            lv_dim = 0

        if quant_correlation_class_name == "Rough_RBF":
            quant_correlation_class = "RBFKernel"

        if quant_correlation_class_name == "Matern32Kernel":
            quant_correlation_class = "Matern32Kernel"

        if quant_correlation_class_name == "Matern52Kernel":
            quant_correlation_class = "Matern52Kernel"

        if quant_correlation_class_name == "Matern12Kernel":
            quant_correlation_class = "Matern12Kernel"

        if len(qual_index) > 0:
            ####################### Defined multiple kernels for seperate variables ###################
            qual_kernels = []
            for i in range(len(qual_kernel_columns)):
                qual_kernels.append(
                    kernels.RBFKernel(active_dims=torch.arange(lv_dim) + lv_dim * i)
                )
                qual_kernels[i].initialize(**{"lengthscale": 1.0})
                qual_kernels[i].raw_lengthscale.requires_grad_(False)

        if len(quant_index) == 0:
            correlation_kernel = qual_kernels[0]
            for i in range(1, len(qual_kernels)):
                correlation_kernel *= qual_kernels[i]
        else:
            try:
                quant_correlation_class = getattr(kernels, quant_correlation_class)
            except:
                raise RuntimeError("%s not an allowed kernel" % quant_correlation_class)

            if quant_correlation_class_name == "RBFKernel":
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns) * lv_dim
                    + torch.arange(len(quant_index)),
                    lengthscale_constraint=Positive(
                        transform=torch.exp, inv_transform=torch.log
                    ),
                )
            elif quant_correlation_class_name == "Rough_RBF":
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns) * lv_dim
                    + torch.arange(len(quant_index)),
                    lengthscale_constraint=Positive(
                        transform=lambda x: 2.0 ** (-0.5) * torch.pow(10, -x / 2),
                        inv_transform=lambda x: -2.0 * torch.log10(x / 2.0),
                    ),
                )

            elif quant_correlation_class_name == "Matern12Kernel":
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns) * lv_dim
                    + torch.arange(len(quant_index)),
                    lengthscale_constraint=Positive(
                        transform=lambda x: 2.0 ** (-0.5) * torch.pow(10, -x / 2),
                        inv_transform=lambda x: -2.0 * torch.log10(x / 2.0),
                    ),
                )

            elif quant_correlation_class_name == "Matern32Kernel":
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns) * lv_dim
                    + torch.arange(len(quant_index)),
                    # lengthscale_constraint= Positive(transform= torch.exp,inv_transform= torch.log)
                    lengthscale_constraint=Positive(
                        transform=lambda x: 2.0 ** (-0.5) * torch.pow(10, -x / 2),
                        inv_transform=lambda x: -2.0 * torch.log10(x / 2.0),
                    ),
                )

            elif quant_correlation_class_name == "Matern52Kernel":
                quant_kernel = quant_correlation_class(
                    ard_num_dims=len(quant_index),
                    active_dims=len(qual_kernel_columns) * lv_dim
                    + torch.arange(len(quant_index)),
                    # lengthscale_constraint= Positive(transform= torch.exp,inv_transform= torch.log)
                    lengthscale_constraint=Positive(
                        transform=lambda x: 2.0 ** (-0.5) * torch.pow(10, -x / 2),
                        inv_transform=lambda x: -2.0 * torch.log10(x / 2.0),
                    ),
                )

            if quant_correlation_class_name == "RBFKernel":

                quant_kernel.register_prior(
                    "lengthscale_prior",
                    MollifiedUniformPrior(math.log(0.1), math.log(10)),
                    "raw_lengthscale",
                )

            elif quant_correlation_class_name == "Rough_RBF":
                quant_kernel.register_prior(
                    "lengthscale_prior", NormalPrior(-3.0, 3.0), "raw_lengthscale"
                )

            elif quant_correlation_class_name == "Matern12Kernel":
                quant_kernel.register_prior(
                    #'lengthscale_prior', MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                    "lengthscale_prior",
                    NormalPrior(-3.0, 3.0),
                    "raw_lengthscale",
                )

            elif quant_correlation_class_name == "Matern32Kernel":
                quant_kernel.register_prior(
                    #'lengthscale_prior', MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                    "lengthscale_prior",
                    NormalPrior(-3.0, 3.0),
                    "raw_lengthscale",
                )

            elif quant_correlation_class_name == "Matern52Kernel":
                quant_kernel.register_prior(
                    #'lengthscale_prior', MollifiedUniformPrior(math.log(0.1),math.log(10)),'raw_lengthscale'
                    "lengthscale_prior",
                    NormalPrior(-3.0, 3.0),
                    "raw_lengthscale",
                )

            if len(qual_index) > 0:
                temp = qual_kernels[0]
                for i in range(1, len(qual_kernels)):
                    temp *= qual_kernels[i]
                correlation_kernel = temp * quant_kernel  # + qual_kernel + quant_kernel
            else:
                correlation_kernel = quant_kernel

        super(GPPLUS, self).__init__(
            train_x=train_x,
            train_y=train_y,
            noise_indices=noise_indices,
            correlation_kernel=correlation_kernel,
            noise=noise,
            fix_noise=fix_noise,
            lb_noise=lb_noise,
        )

        # register index and transforms
        self.register_buffer("quant_index", torch.tensor(quant_index))
        self.register_buffer("qual_index", torch.tensor(qual_index))

        self.num_levels_per_var = num_levels_per_var
        self.lv_dim = lv_dim
        self.uniform_encoding_columns = uniform_encoding_columns
        self.encoding_type = encoding_type
        self.perm = []
        self.zeta = []
        self.perm_dict = []
        self.A_matrix = []
        self.collocation_x = collocation_x  ####### ADDED
        self.alpha = 1.0
        self.beta = 20.0
        self.covar_inv = None
        self.omega = 3  # 3.2
        self.name_output = name_output
        self.chol_decomp = None
        self.g_uvp = None
        self.k_xX = None

        if len(qual_kernel_columns) > 0:
            for i in range(len(qual_kernel_columns)):
                if type(qual_kernel_columns[i]) == int:
                    num = self.num_levels_per_var[
                        qual_index.index(qual_kernel_columns[i])
                    ]
                    cat = [num]
                else:
                    cat = [
                        self.num_levels_per_var[qual_index.index(k)]
                        for k in qual_kernel_columns[i]
                    ]
                    num = sum(cat)

                zeta, perm, perm_dict = self.zeta_matrix(
                    num_levels=cat, lv_dim=self.lv_dim
                )
                self.zeta.append(zeta.to(**tkwargs))
                self.perm.append(perm.to(**tkwargs))
                self.perm_dict.append(perm_dict)

                model_temp = FFNN(
                    self,
                    input_size=num,
                    num_classes=lv_dim,
                    layers=NN_layers,
                    name=str(qual_kernel_columns[i]),
                ).to(**tkwargs)
                self.A_matrix.append(model_temp.to(**tkwargs))

        self.basis = basis
        i = 0
        if self.basis == "single":
            self.mean_module = gpytorch.means.ConstantMean(prior=NormalPrior(0.0, 1.0))
            self.mean_module.constant.data = torch.tensor(
                [0.0]
            )  # Set the desired value
            self.mean_module.constant.requires_grad = False  # Fix the hyperparameter
        elif self.basis == "multiple_constant":
            if basis_function_size is None:
                basis_function_size = train_x.shape[1] - 1
            self.num_sources = int(torch.max(train_x[:, -1]))
            for i in range(self.num_sources + 1):
                if i == 0:
                    setattr(self, "mean_module_" + str(i), gpytorch.means.ZeroMean())

                else:
                    # Constant
                    setattr(
                        self,
                        "mean_module_" + str(i),
                        gpytorch.means.ConstantMean(prior=NormalPrior(0.0, 0.3)),
                    )

        elif self.basis == "multiple_polynomial":
            if basis_function_size is None:
                basis_function_size = train_x.shape[1] - 1
            self.num_sources = int(torch.max(train_x[:, -1]))
            for i in range(self.num_sources + 1):
                if i == 0:
                    setattr(self, "mean_module_" + str(i), gpytorch.means.ZeroMean())
                else:
                    setattr(
                        self,
                        "mean_module_" + str(i),
                        LinearMean_with_prior(
                            input_size=basis_function_size,
                            batch_shape=torch.Size([]),
                            bias=True,
                        ),
                    )
        elif self.basis == "neural_network":
            ############################################### One NN for ALL
            if len(qual_index) == 0:
                setattr(
                    self,
                    "mean_module_NN_All",
                    FFNN_for_Mean(
                        self,
                        input_size=train_x.shape[1],
                        num_classes=4,
                        layers=NN_layers_base,
                        name=str("mean_module_" + str(i) + "_"),
                    ),
                )
            else:
                setattr(
                    self,
                    "mean_module_NN_All",
                    FFNN_for_Mean(
                        self,
                        input_size=train_x.shape[1] - len(qual_index) + 2,
                        num_classes=1,
                        layers=NN_layers_base,
                        name=str("mean_module_" + str(i) + "_"),
                    ),
                )
        elif self.basis == "M3":
            setattr(
                self,
                "mean_module_NN_All",
                NetworkM4(
                    input_dim=train_x.shape[1], output_dim=3, layers=NN_layers_base
                ),
            )

        # Fix the hyperparameter value
        self.covar_module.base_kernel.raw_lengthscale.data = torch.tensor(
            [self.omega, self.omega], dtype=torch.float32
        )  # Set the desired value
        self.covar_module.base_kernel.raw_lengthscale.requires_grad = (
            False  # Fix the hyperparameter
        )

        self.covar_module.raw_outputscale.data = torch.tensor(
            0.541
        )  # Set the desired value
        self.covar_module.raw_outputscale.requires_grad = (
            False  # Fix the hyperparameter
        )

    def forward(self, x: torch.Tensor) -> MultivariateNormal:
        x_forward_raw = x.clone()
        nd_flag = 0
        if x.dim() > 2:
            xsize = x.shape
            x = x.reshape(-1, x.shape[-1])
            nd_flag = 1

        if len(self.qual_kernel_columns) > 0:
            embeddings = []
            for i in range(len(self.qual_kernel_columns)):
                temp = self.transform_categorical(
                    x=x[:, self.qual_kernel_columns[i]].clone().type(torch.int64),
                    perm_dict=self.perm_dict[i],
                    zeta=self.zeta[i],
                )
                embeddings.append(self.A_matrix[i](temp))

            x = torch.cat([embeddings[0], x[..., self.quant_index]], dim=-1)

        if nd_flag == 1:
            x = x.reshape(*xsize[:-1], -1)

        #################### Multiple bases (General Case) ####################################
        def multi_mean(x, x_forward_raw):
            mean_x = torch.zeros_like(x[:, -1])
            if self.basis == "single":
                mean_x = self.mean_module(x)
            elif self.basis == "multiple_constant":
                for i in range(len(mean_x)):
                    qq = int(x_forward_raw[i, -1])
                    mean_x[i] = getattr(self, "mean_module_" + str(qq))(
                        torch.tensor(x[i, -1].clone()).reshape(-1, 1)
                    )
            elif self.basis == "multiple_polynomial":
                for i in range(len(mean_x)):
                    qq = int(x_forward_raw[i, -1])
                    mean_x[i] = getattr(self, "mean_module_" + str(qq))(
                        torch.cat(
                            (
                                torch.tensor(
                                    (x[i, -1].clone().double().reshape(-1, 1)) ** 2
                                ),
                                torch.tensor(x[i, -1].clone().double()).reshape(-1, 1),
                            ),
                            1,
                        )
                    )

            elif self.basis == "neural_network":
                mean_x = getattr(self, "mean_module_NN_All")(
                    x.clone()
                )  # .reshape(-1) #### FOR MULTIOUTPUT DELETE RESHAPE

            elif self.basis == "M3":
                if hasattr(self, "name_output"):
                    mean_x = getattr(self, "mean_module_NN_All")(x.clone())
                else:
                    mean_x = getattr(self, "mean_module_NN_All")(x.clone()).reshape(-1)

            return mean_x

        ##########################################################################################
        if self.name_output == "u":
            mean_x = multi_mean(x, x_forward_raw)[:, 0].reshape(-1)
        if self.name_output == "v":
            mean_x = multi_mean(x, x_forward_raw)[:, 1].reshape(-1)
        if self.name_output == "p":
            mean_x = multi_mean(x, x_forward_raw)[:, 2].reshape(-1)
        if self.name_output == "ro":
            mean_x = multi_mean(x, x_forward_raw)[:, 3].reshape(-1)
        covar_x = self.covar_module(x)

        return MultivariateNormal(mean_x, covar_x)

    def predict(self, Xtest, return_std=True, include_noise=True):
        with torch.no_grad():
            return super().predict(
                Xtest, return_std=return_std, include_noise=include_noise
            )

    def predict_with_grad(self, Xtest, return_std=True, include_noise=True):
        return super().predict(
            Xtest, return_std=return_std, include_noise=include_noise
        )

    def noise_value(self):
        noise = self.likelihood.noise_covar.noise.detach() * self.y_std**2
        return noise

    def visualize_latent(self, suptitle=None):
        if len(self.qual_kernel_columns) > 0:
            for i in range(len(self.qual_kernel_columns)):
                zeta = self.zeta[i]
                A = self.A_matrix[i]

                positions = A(zeta)
                level = torch.max(self.perm[i], axis=0)[0].tolist()
                perm = self.perm[i]
                plot_sep(
                    positions=positions,
                    levels=level,
                    perm=perm,
                    constraints_flag=True,
                )

    def visualize_latent_position(self, lv_columns=None):
        if len(self.qual_kernel_columns) > 0:
            for i in range(len(self.qual_kernel_columns)):
                zeta = self.zeta[i]
                A = self.A_matrix[i]
                positions = A(zeta)
                if self.qual_kernel_columns[i] == lv_columns[0]:
                    return positions

    def visualize_latent_position_simple(self, suptitle=None):
        if len(self.qual_kernel_columns) > 0:
            for i in range(len(self.qual_kernel_columns)):
                zeta = self.zeta[i]
                A = self.A_matrix[i]
                positions = A(zeta)
        return positions

    @classmethod
    def show(cls):
        plt.show()

    def get_params(self, name=None):
        params = {}
        print("###################Parameters###########################")
        for n, value in self.named_parameters():
            params[n] = value
        if name is None:
            print(params)
            return params
        else:
            if name == "Mean":
                key = "mean_module.constant"
            elif name == "Sigma":
                key = "covar_module.raw_outputscale"
            elif name == "Noise":
                key = "likelihood.noise_covar.raw_noise"
            elif name == "Omega":
                for n in params.keys():
                    if "raw_lengthscale" in n and params[n].numel() > 1:
                        key = n
            print(params[key])
            return params[key]

    def get_latent_space(self):
        if len(self.qual_index) > 0:
            zeta = torch.tensor(self.zeta)
            positions = self.nn_model(zeta)
            return positions.detach()
        else:
            print("No categorical Variable, No latent positions")
            return None

    def zeta_matrix(
        self, num_levels: int, lv_dim: int, batch_shape=torch.Size()
    ) -> None:

        if any([i == 1 for i in num_levels]):
            raise ValueError("Categorical variable has only one level!")

        if lv_dim == 1:
            raise RuntimeWarning("1D latent variables are difficult to optimize!")

        for level in num_levels:
            if lv_dim > level - 0:
                lv_dim = min(lv_dim, level - 1)
                raise RuntimeWarning(
                    "The LV dimension can atmost be num_levels-1. "
                    "Setting it to %s in place of %s" % (level - 1, lv_dim)
                )

        from itertools import product

        levels = []
        for l in num_levels:
            levels.append(torch.arange(l))

        perm = list(product(*levels))
        perm = torch.tensor(perm, dtype=torch.int64)

        # -------------Mapping-------------------------
        perm_dic = {}
        for i, row in enumerate(perm):
            temp = str(row.tolist())
            if temp not in perm_dic.keys():
                perm_dic[temp] = i

        # -------------One_hot_encoding------------------
        for ii in range(perm.shape[-1]):
            if perm[..., ii].min() != 0:
                perm[..., ii] -= perm[..., ii].min()

        perm_one_hot = []
        for i in range(perm.size()[1]):
            perm_one_hot.append(torch.nn.functional.one_hot(perm[:, i]))

        perm_one_hot = torch.concat(perm_one_hot, axis=1)

        return perm_one_hot, perm, perm_dic

    def transform_categorical(self, x: torch.Tensor, perm_dict=[], zeta=[]) -> None:
        if x.dim() == 1:
            x = x.reshape(-1, 1)
        # categorical should start from 0
        if self.training == False:
            x = setlevels(x.cpu())
        if self.encoding_type == "one-hot":
            index = [perm_dict[str(row.tolist())] for row in x]

            if x.dim() == 1:
                x = x.reshape(
                    len(x),
                )

            return zeta[index, :]

        elif self.encoding_type == "uniform":

            temp2 = np.random.uniform(
                0, 1, (len(self.perm), self.uniform_encoding_columns)
            )
            dict = {}
            dict2 = {}

            for i in range(0, self.perm.shape[0]):
                dict[tuple((self.perm[i, :]).numpy())] = temp2[i, :]

            for i in range(0, x.shape[0]):
                dict2[i] = dict[tuple((x[i]).numpy())]

            x_one_hot = torch.from_numpy(np.array(list(dict2.values())))
        else:
            raise ValueError("Invalid type")

        return x_one_hot


##########################################################################################################################################################


class LinearMean_with_prior(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.register_parameter(
            name="weights",
            parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)),
        )
        self.register_prior(
            name="weights_prior",
            prior=gpytorch.priors.NormalPrior(0.0, 1.0),
            param_or_closure="weights",
        )

        if bias:
            self.register_parameter(
                name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1))
            )
            self.register_prior(
                name="bias_prior",
                prior=gpytorch.priors.NormalPrior(0.0, 1.0),
                param_or_closure="bias",
            )

        else:
            self.bias = None

    def forward(self, x):
        res = x.matmul(self.weights).squeeze(-1)
        if self.bias is not None:
            res = res + self.bias
        return res


###############################################################################################################################################################################################################################################################
class FFNN(torch.nn.Module):
    def __init__(self, lmgp, input_size, num_classes, layers, name):
        super(FFNN, self).__init__()
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.hidden_num = len(layers)
        if self.hidden_num > 0:
            self.fci = torch.nn.Linear(input_size, layers[0], bias=False)
            lmgp.register_parameter("fci", self.fci.weight)
            lmgp.register_prior(
                name="latent_prior_fci",
                prior=gpytorch.priors.NormalPrior(0.0, 3.0),
                param_or_closure="fci",
            )

            for i in range(1, self.hidden_num):
                # self.h = nn.Linear(neuran[i-1], neuran[i])
                setattr(
                    self,
                    "h" + str(i),
                    torch.nn.Linear(layers[i - 1], layers[i], bias=False),
                )
                lmgp.register_parameter(
                    "h" + str(i), getattr(self, "h" + str(i)).weight
                )
                lmgp.register_prior(
                    name="latent_prior" + str(i),
                    prior=gpytorch.priors.NormalPrior(0.0, 3.0),
                    param_or_closure="h" + str(i),
                )

            self.fce = torch.nn.Linear(layers[-1], num_classes, bias=False)
            lmgp.register_parameter("fce", self.fce.weight)
            lmgp.register_prior(
                name="latent_prior_fce",
                prior=gpytorch.priors.NormalPrior(0.0, 3.0),
                param_or_closure="fce",
            )
        else:
            self.fci = Linear_MAP(input_size, num_classes, bias=False)
            lmgp.register_parameter(name, self.fci.weight)
            lmgp.register_prior(
                name="latent_prior_" + name,
                prior=gpytorch.priors.NormalPrior(0, 3),
                param_or_closure=name,
            )

    def forward(self, x, transform=lambda x: x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """
        if self.hidden_num > 0:
            x = torch.tanh(self.fci(x))
            for i in range(1, self.hidden_num):
                x = torch.tanh(getattr(self, "h" + str(i))(x))

            x = self.fce(x)
        else:
            x = self.fci(x, transform)
        return x


class FFNN_for_Mean(gpytorch.Module):
    def __init__(self, lmgp, input_size, num_classes, layers, name):
        super(FFNN_for_Mean, self).__init__()
        self.dropout = torch.nn.Dropout(0.0)
        # Our first linear layer take input_size, in this case 784 nodes to 50
        # and our second linear layer takes 50 to the num_classes we have, in
        # this case 10.
        self.hidden_num = len(layers)
        if self.hidden_num > 0:
            self.fci = Linear_new(input_size, layers[0], bias=True, name="fci")
            for i in range(1, self.hidden_num):
                setattr(
                    self,
                    "h" + str(i),
                    Linear_new(layers[i - 1], layers[i], bias=True, name="h" + str(i)),
                )

            self.fce = Linear_new(layers[-1], num_classes, bias=True, name="fce")
        else:
            self.fci = Linear_new(
                input_size, num_classes, bias=True, name="fci"
            )  # Linear_MAP(input_size, num_classes, bias = True)

    def forward(self, x, transform=lambda x: x):
        """
        x here is the mnist images and we run it through fc1, fc2 that we created above.
        we also add a ReLU activation function in between and for that (since it has no parameters)
        I recommend using nn.functional (F)
        """
        if self.hidden_num > 0:

            x = torch.tanh(self.fci(x))
            # x = self.dropout(x)
            # x = self.fci(x)
            for i in range(1, self.hidden_num):
                x = torch.tanh(getattr(self, "h" + str(i))(x))

            x = self.fce(x)
            # x = torch.cat([x[...,:3] , torch.sigmoid(x[...,3]).unsqueeze(-1)] , dim = -1)
        else:
            x = self.fci(x)
            # x = torch.cat([x[...,:3] , torch.tanh(x[...,3]).unsqueeze(-1)] , dim = -1)

        return x


class NetworkM4(torch.nn.Module):
    def __init__(
        self,
        input_dim=2,
        output_dim=1,
        layers=[40, 40, 40, 40],
        activation="tanh",
        collocation_x=[],
    ) -> None:
        super(NetworkM4, self).__init__()
        activation_list = {
            "tanh": torch.nn.Tanh(),
            "Silu": torch.nn.SiLU(),
            "Sigmoid": torch.nn.Sigmoid(),
        }
        activation = activation_list[activation]
        self.dim = layers[0]

        self.U = torch.nn.Linear(input_dim, self.dim).to("cuda")
        self.V = torch.nn.Linear(input_dim, self.dim).to("cuda")
        self.H1 = torch.nn.Linear(input_dim, self.dim).to("cuda")
        self.last = torch.nn.Linear(self.dim, output_dim).to("cuda")

        self.collocation_x = collocation_x
        self.alpha = 1.0
        self.beta = 1.0

        l = torch.nn.ModuleList()
        for _ in range(len(layers)):
            l.append(torch.nn.Linear(self.dim, self.dim))
            l.append(activation)
        self.layers = torch.nn.Sequential(*l).to("cuda")

    def forward(self, input):
        U = torch.nn.Tanh()(self.U(input))
        V = torch.nn.Tanh()(self.V(input))
        H = torch.nn.Tanh()(self.H1(input))

        for layer in self.layers:
            Z = layer(H)
            H = (1 - Z) * U + Z * V

        out = self.last(H)
        return out


class Linear_new(gpytorch.means.Mean):
    r"""Applies a linear transformation to the incoming data: :math:`y = xA^T + b`

    This module supports :ref:`TensorFloat32<tf32_on_ampere>`.

    On certain ROCm devices, when using float16 inputs this module will use :ref:`different precision<fp16_on_mi200>` for backward.

    Args:
        in_features: size of each input sample
        out_features: size of each output sample
        bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``

    Shape:
        - Input: :math:`(*, H_{in})` where :math:`*` means any number of
          dimensions including none and :math:`H_{in} = \text{in\_features}`.
        - Output: :math:`(*, H_{out})` where all but the last dimension
          are the same shape as the input and :math:`H_{out} = \text{out\_features}`.

    Attributes:
        weight: the learnable weights of the module of shape
            :math:`(\text{out\_features}, \text{in\_features})`. The values are
            initialized from :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})`, where
            :math:`k = \frac{1}{\text{in\_features}}`
        bias:   the learnable bias of the module of shape :math:`(\text{out\_features})`.
                If :attr:`bias` is ``True``, the values are initialized from
                :math:`\mathcal{U}(-\sqrt{k}, \sqrt{k})` where
                :math:`k = \frac{1}{\text{in\_features}}`

    Examples::

        >>> m = nn.Linear(20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
        torch.Size([128, 30])
    """

    __constants__ = ["in_features", "out_features"]
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name=None,
        device=None,
        dtype=None,
    ) -> None:
        # factory_kwargs = {'device': device, 'dtype': dtype}
        # factory_kwargs=tkwargs
        super(Linear_new, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.name = str(name)
        self.register_parameter(
            name=str(self.name) + "weight",
            parameter=Parameter(torch.empty((out_features, in_features))),
        )
        self.register_prior(
            name=str(self.name) + "prior_m_weight_fci",
            prior=gpytorch.priors.NormalPrior(0.0, 1.0),
            param_or_closure=str(self.name) + "weight",
        )

        if bias:
            self.register_parameter(
                name=str(self.name) + "bias",
                parameter=Parameter(torch.empty(out_features)),
            )
            self.register_prior(
                name=str(self.name) + "prior_m_bias_fci",
                prior=gpytorch.priors.NormalPrior(0.0, 1.0),
                param_or_closure=str(self.name) + "bias",
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        torch.nn.init.kaiming_uniform_(
            getattr(self, str(self.name) + "weight"), a=math.sqrt(5)
        )
        if getattr(self, str(self.name) + "bias") is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(
                getattr(self, str(self.name) + "weight")
            )
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            torch.nn.init.uniform_(
                getattr(self, str(self.name) + "bias"), -bound, bound
            )

    def forward(self, input) -> torch.Tensor:
        # return F.linear(input, self.weight, self.bias)
        # print(getattr(self,str(self.name)+'weight'))

        # return F.linear(input, getattr(self,str(self.name)+'weight').double(), getattr(self,str(self.name)+'bias').double())      ### Forced to Add .double() for NN in mean function
        return F.linear(
            input,
            getattr(self, str(self.name) + "weight"),
            getattr(self, str(self.name) + "bias"),
        )  ### Forced to Add .double() for NN in mean function

    def extra_repr(self) -> str:
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class Linear_MAP(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(in_features, out_features, bias, device, dtype)

    def forward(self, input, transform=lambda x: x):
        return F.linear(input, transform(self.weight), self.bias)
