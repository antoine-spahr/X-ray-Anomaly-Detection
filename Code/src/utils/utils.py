from prettytable import PrettyTable

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

    """
    summary = PrettyTable([index_header] + list(list(d.values())[0].keys()))
    for key, val in d.items():
        summary.add_row([key] + list(val.values()))

    print(summary)
