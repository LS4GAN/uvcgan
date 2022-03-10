from torch.utils.data.dataloader import default_collate

def collate_imbalanced_seq(seq):
    result = [ x for x in seq if x is not None ]
    if len(result) == 0:
        return None

    return default_collate(result)

def imbalanced_collate(batch):
    elem = batch[0]

    if isinstance(elem, (tuple, list)):
        elem_type = type(elem)
        return elem_type(collate_imbalanced_seq(x) for x in zip(*batch))

    if isinstance(elem, dict):
        return { k : imbalanced_collate([v[k] for v in batch]) for k in elem }

    return default_collate(batch)

