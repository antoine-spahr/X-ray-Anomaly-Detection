from prettytable import PrettyTable
import torch
import torch.nn as nn
from torch.autograd import Variable

from collections import OrderedDict
import numpy as np

def print_progessbar(N, Max, Name='', Size=10, end_char=''):
    """
    Print a progress bar. To be used in a for-loop and called at each iteration
    with the iteration number and the max number of iteration.
    ------------
    INPUT
        |---- N (int) the iteration current number
        |---- Max (int) the total number of iteration
        |---- Name (str) an optional name for the progress bar
        |---- Size (int) the size of the progress bar
        |---- end_char (str) the print end parameter to used in the end of the
        |                    of the progress bar (default is '')
    OUTPUT
        |---- None
    """
    print(f'\r{Name} {N+1:03d}/{Max:03d}'.ljust(len(Name) + 10) \
        + f'[{"#"*int(Size*(N+1)/Max)}'.ljust(Size+1) + f'] {(int(100*(N+1)/Max))}%'.ljust(6), \
        end=end_char)

    if N+1 == Max:
        print('')

def print_param_summary(**params):
    """
    Print the dictionnary passed as a table.
    ------------
    INPUT
        |---- params (keyword arguments) value to display
    OUTPUT
        |---- None
    """
    # get the max length of values and keys
    max_len = max([len(str(key)) for key in params.keys()])+5
    max_len_val = max([max([len(subval) for subval in str(val).split('\n')]) for val in params.values()])+3
    # print header
    print('-'*(max_len+max_len_val+1))
    print('| Parameter'.ljust(max_len) + '| Value'.ljust(max_len_val)+'|')
    print('-'*(max_len+max_len_val+1))
    # print values and subvalues
    for key, value in params.items():
        for i, subvalue in enumerate(str(value).split('\n')):
            if i == 0 :
                print(f'| {key}'.ljust(max_len)+f'| {subvalue}'.ljust(max_len_val)+'|')
            else :
                print('| '.ljust(max_len)+f'| {subvalue}'.ljust(max_len_val)+'|')
    print('-'*(max_len+max_len_val+1))

def print_summary_from_dict(d, index_header=' '):
    """
    Print the dictionnary of doctionnaries passed as a table.
    ------------
    INPUT
        |---- d (dict) the dict of dict to display.
        |---- index_header (str) the name of the index column
    OUTPUT
        |---- None
    """
    summary = PrettyTable([index_header] + list(list(d.values())[0].keys()))
    for key, val in d.items():
        summary.add_row([key] + list(val.values()))

    print(summary)

def summary_string(model, input_size, batch_size=-1, device=torch.device('cuda:0'), dtypes=None):
    """
    Print a summary of the Pytorch network architecture.
    (https://github.com/sksq96/pytorch-summary/blob/master/torchsummary/torchsummary.py)
    ------------
    INPUT
        |---- model (nn.Module) the network to summaries.
        |---- input_size (tuple) the input tensor dimension of the network
        |           (without the batch size).
        |---- batch_size (int) the batch size to display.
        |---- device (str) the device on which to work.
        |---- dtypes (str) the dtype of the input tensor.
    OUTPUT
        |---- summary_string (str) the summary table of the network architecture
        |           as a string.
    """
    if dtypes == None:
        dtypes = [torch.FloatTensor]*len(input_size)

    summary_str = ''

    def register_hook(module):
        def hook(module, input, output):
            class_name = str(module.__class__).split(".")[-1].split("'")[0]
            module_idx = len(summary)

            m_key = "%s-%i" % (class_name, module_idx + 1)
            summary[m_key] = OrderedDict()
            summary[m_key]["input_shape"] = list(input[0].size())
            summary[m_key]["input_shape"][0] = batch_size
            if isinstance(output, (list, tuple)):
                summary[m_key]["output_shape"] = [
                    [-1] + list(o.size())[1:] for o in output
                ]
            else:
                summary[m_key]["output_shape"] = list(output.size())
                summary[m_key]["output_shape"][0] = batch_size

            params = 0
            if hasattr(module, "weight") and hasattr(module.weight, "size"):
                params += torch.prod(torch.LongTensor(list(module.weight.size())))
                summary[m_key]["trainable"] = module.weight.requires_grad
            if hasattr(module, "bias") and hasattr(module.bias, "size"):
                params += torch.prod(torch.LongTensor(list(module.bias.size())))
            summary[m_key]["nb_params"] = params

        if (
            not isinstance(module, nn.Sequential)
            and not isinstance(module, nn.ModuleList)
        ):
            hooks.append(module.register_forward_hook(hook))

    # multiple inputs to the network
    if isinstance(input_size, tuple):
        input_size = [input_size]

    # batch_size of 2 for batchnorm
    x = [torch.rand(2, *in_size).type(dtype).to(device=device)
         for in_size, dtype in zip(input_size, dtypes)]

    # create properties
    summary = OrderedDict()
    hooks = []

    # register hook
    model.apply(register_hook)

    # make a forward pass
    # print(x.shape)
    model(*x)

    # remove these hooks
    for h in hooks:
        h.remove()

    summary_str += "----------------------------------------------------------------" + "\n"
    line_new = "{:>20}  {:>25} {:>15}".format(
        "Layer (type)", "Output Shape", "Param #")
    summary_str += line_new + "\n"
    summary_str += "================================================================" + "\n"
    total_params = 0
    total_output = 0
    trainable_params = 0
    for layer in summary:
        # input_shape, output_shape, trainable, nb_params
        line_new = "{:>20}  {:>25} {:>15}".format(
            layer,
            str(summary[layer]["output_shape"]),
            "{0:,}".format(summary[layer]["nb_params"]),
        )
        total_params += summary[layer]["nb_params"]

        total_output += np.prod(summary[layer]["output_shape"])
        if "trainable" in summary[layer]:
            if summary[layer]["trainable"] == True:
                trainable_params += summary[layer]["nb_params"]
        summary_str += line_new + "\n"

    # assume 4 bytes/number (float on cuda).
    total_input_size = abs(np.prod(sum(input_size, ()))
                           * batch_size * 4. / (1024 ** 2.))
    total_output_size = abs(2. * total_output * 4. /
                            (1024 ** 2.))  # x2 for gradients
    total_params_size = abs(total_params * 4. / (1024 ** 2.))
    total_size = total_params_size + total_output_size + total_input_size

    summary_str += "================================================================" + "\n"
    summary_str += "Total params: {0:,}".format(total_params) + "\n"
    summary_str += "Trainable params: {0:,}".format(trainable_params) + "\n"
    summary_str += "Non-trainable params: {0:,}".format(total_params -
                                                        trainable_params) + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    summary_str += "Input size (MB): %0.2f" % total_input_size + "\n"
    summary_str += "Forward/backward pass size (MB): %0.2f" % total_output_size + "\n"
    summary_str += "Params size (MB): %0.2f" % total_params_size + "\n"
    summary_str += "Estimated Total Size (MB): %0.2f" % total_size + "\n"
    summary_str += "----------------------------------------------------------------" + "\n"
    # return summary
    return summary_str#, (total_params, trainable_params)
