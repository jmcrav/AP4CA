from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

def get_dataloader(input_ids,
                   attention_mask,
                   labels_actions,
                   labels_attributes,
                   dialog_ids,
                   turn_idx,
                   batch_size,
                   type='sequential'):
    """

    :param input_ids:
    :param attention_mask:
    :param labels_actions:
    :param labels_attributes:
    :param dialog_ids:
    :param turn_idx:
    :param type:
    :return:
    """

    dataset = TensorDataset(input_ids, attention_mask, labels_actions, labels_attributes, dialog_ids, turn_idx)

    if type=='sequential':
        return DataLoader(dataset, sampler = SequentialSampler(dataset), batch_size = batch_size)
    elif type=='random':
        return DataLoader(dataset, sampler = RandomSampler(dataset), batch_size = batch_size)
    else:
        raise ValueError(f"Type dataloader '{type}' invalid, must be 'random' or 'sequential'")
