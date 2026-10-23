import numpy as np

def windows(features, actions, *, context, targets, action_sizes, task_id: int):
  t = len(features)
  factors = len(action_sizes)
  past_dim = sum(action_sizes)
  out = []
  for start in range(0, t, context):
    stop = min(start + context, t)
    valid_length = stop - start
    feats = np.zeros((context, features.shape[1]), dtype=np.float32)
    past = np.zeros((context, past_dim), dtype=np.float32)
    tasks = np.full(context, int(task_id), dtype=np.int64)
    labels = np.zeros((context, targets, factors), dtype=np.int64)
    valid = np.zeros((context, targets), dtype=bool)
    feats[:valid_length] = features[start:stop]
    for i in range(valid_length):
      pos = start + i
      if pos > 0:
        offset = 0
        for k, n in enumerate(action_sizes):
          past[i, offset + int(actions[pos - 1, k])] = 1
          offset += n
      for horizon in range(targets):
        target = pos + horizon
        if target >= t:
          break
        labels[i, horizon] = actions[target]
        valid[i, horizon] = True
    if not valid.any():
      continue
    out.append(dict(features=feats, past_actions=past, tasks=tasks, targets=labels, valid=valid))
  return out
