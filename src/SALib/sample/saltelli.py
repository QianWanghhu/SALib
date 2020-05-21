from __future__ import division

import numpy as np

from . import common_args
from . import sobol_sequence, latin
from ..util import scale_samples, nonuniform_scale_samples, read_param_file, compute_groups_matrix


def sample(problem, N, calc_second_order=True, skip_values=1000, deleted=True,
            product_dist=None, problem_adjust=None, seed=None):
    """Generates model inputs using Saltelli's extension of the Sobol sequence.

    Returns a NumPy matrix containing the model inputs using Saltelli's sampling
    scheme.  Saltelli's scheme extends the Sobol sequence in a way to reduce
    the error rates in the resulting sensitivity index calculations.  If
    calc_second_order is False, the resulting matrix has N * (D + 2)
    rows, where D is the number of parameters.  If calc_second_order is True,
    the resulting matrix has N * (2D + 2) rows.  These model inputs are
    intended to be used with :func:`SALib.analyze.sobol.analyze`.

    Parameters
    ----------
    problem : dict
        The problem definition
    N : int
        The number of samples to generate
    calc_second_order : bool
        Calculate second-order sensitivities (default True)
    skip_values : int
        The number of samples to skip (default 1000)
    deleted : bool
        whether to delete columns which are used to calculate the product of other parameters
        (default True), if False, set values in those columns as 1
    product_dist : np.ndarray
        Array for identifying parameters which are going to be compressed.
        Each row reprenst a group of parameters to compress and the first column is the targeted product column
        If none, treated as original Saltelli sampling,  problem_adjust must not be given.
    problem_adjust : dict
        dictionary for the new problem       
    """
    if seed:
        np.random.seed(seed)

    D = problem['num_vars']
    groups = problem.get('groups')

    if not groups:
        Dg = problem['num_vars']
    else:
        Dg = len(set(groups))
        G, group_names = compute_groups_matrix(groups)

    # How many values of the Sobol sequence to skip  
    skip_values = skip_values

    # Create base sequence - could be any type of sampling
    base_sequence = sobol_sequence.sample(N + skip_values, 2 * D)
    # Create base sequence - using LHS
    # base_sequence = latin.sample(problem, N+skip_values, scale=False, seed=seed)
    # re-scale sampline
    base_sequence = np.vstack([base_sequence[:, 0:D], base_sequence[:, D:]])

    if not problem.get('dists'):
        # scaling values out of 0-1 range with uniform distributions
        scale_samples(base_sequence, problem['bounds'])
    else:
        # scaling values to other distributions based on inverse CDFs
        base_sequence = nonuniform_scale_samples(
            base_sequence, problem['bounds'], problem['dists'])
    base_sequence = np.hstack([base_sequence[0: (N + skip_values), :], 
                            base_sequence[(N + skip_values):, :]])

    if isinstance(product_dist, np.ndarray):
        column_delete = np.array([])
        for index_list in product_dist:
            second_part = np.array([jj+D for jj in index_list])
            base_sequence[:, index_list[0]] = np.prod(base_sequence[:, index_list], axis=1)
            base_sequence[:, second_part[0]] = np.prod(base_sequence[:, second_part], axis=1)
            column_delete = np.append(column_delete, index_list[1:])
            column_delete = np.append(column_delete, second_part[1:])
        if not deleted:
            base_sequence[:, column_delete.astype(int)] = [1]
        else:
            base_sequence = np.delete(base_sequence, column_delete, axis=1)
    # re-define problem using the new problem (problem_adjust)
            problem = problem_adjust
            D = problem['num_vars']
            groups = problem.get('groups')

        if not groups:
            Dg = problem['num_vars']
        else:
            Dg = len(set(groups))
            G, group_names = compute_groups_matrix(groups)    
    # End re-define

    if calc_second_order:
        saltelli_sequence = np.zeros([(2 * Dg + 2) * N, D])
    else:
        saltelli_sequence = np.zeros([(Dg + 2) * N, D])
    index = 0

    for i in range(skip_values, N + skip_values):

        # Copy matrix "A"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j]

        index += 1

        # Cross-sample elements of "B" into "A"
        for k in range(Dg):
            for j in range(D):
                if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                    saltelli_sequence[index, j] = base_sequence[i, j + D]
                else:
                    saltelli_sequence[index, j] = base_sequence[i, j]

            index += 1

        # Cross-sample elements of "A" into "B"
        # Only needed if you're doing second-order indices (true by default)
        if calc_second_order:
            for k in range(Dg):
                for j in range(D):
                    if (not groups and j == k) or (groups and group_names[k] == groups[j]):
                        saltelli_sequence[index, j] = base_sequence[i, j]
                    else:
                        saltelli_sequence[index, j] = base_sequence[i, j + D]

                index += 1

        # Copy matrix "B"
        for j in range(D):
            saltelli_sequence[index, j] = base_sequence[i, j + D]

        index += 1
    # if not problem.get('dists'):
    #     # scaling values out of 0-1 range with uniform distributions
    #     scale_samples(saltelli_sequence, problem['bounds'])
    #     return saltelli_sequence
    # else:
    #     # scaling values to other distributions based on inverse CDFs
    #     scaled_saltelli = nonuniform_scale_samples(
    #         saltelli_sequence, problem['bounds'], problem['dists'])
    scaled_saltelli = saltelli_sequence
    return scaled_saltelli


def cli_parse(parser):
    """Add method specific options to CLI parser.

    Parameters
    ----------
    parser : argparse object

    Returns
    ----------
    Updated argparse object
    """
    parser.add_argument('--max-order', type=int, required=False, default=2,
                        choices=[1, 2],
                        help='Maximum order of sensitivity indices \
                           to calculate')
    return parser


def cli_action(args):
    """Run sampling method

    Parameters
    ----------
    args : argparse namespace
    """
    problem = read_param_file(args.paramfile)
    param_values = sample(problem, args.samples,
                          calc_second_order=(args.max_order == 2),
                          seed=args.seed)
    np.savetxt(args.output, param_values, delimiter=args.delimiter,
               fmt='%.' + str(args.precision) + 'e')


if __name__ == "__main__":
    common_args.run_cli(cli_parse, cli_action)
