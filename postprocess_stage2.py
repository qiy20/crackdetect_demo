def postprocess_stage2(pred_file, lower_bound, upper_bound, criterion):
    """

    :param pred_file: the result of stage1
    :param lower_bound:
    :param upper_bound:
    :param criterion: the criterion that are user-defined: 0<=criterion<=1
    :return:
    """
    res = []
    threshold = lower_bound + (upper_bound - lower_bound) * criterion
    with open(pred_file) as f:
        pred = f.readlines()
    for line in pred:
        if float(line.split()[-1]) >= threshold:
            res.append(line)
    return res
