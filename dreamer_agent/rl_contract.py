"""Immutable policy/reward history contract; compatibility is not calibration."""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import torch


def module_sha256(module):
    """Include names/dtypes/shapes and all buffers; independent of checkpoint format."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        value = value.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class HistoryRewardContract:
    context: int
    policy_tasks: tuple[str, ...]
    reward_tasks: tuple[str, ...]
    policy_state_sha256: str
    reward_state_sha256: str
    feature_contract_sha256: str
    action_spec_sha256: str
    history_protocol: str = 'zero_state_blocks_retain_previous_action'
    reward_protocol: str = 'task_origin_reset'
    inference: str = 'incremental_v1'
    schema_version: int = 1

    def __post_init__(self):
        # Normalize JSON lists so a frozen contract cannot contain mutable lists.
        object.__setattr__(self, 'policy_tasks', tuple(self.policy_tasks))
        object.__setattr__(self, 'reward_tasks', tuple(self.reward_tasks))
        if type(self.context) is not int or self.context not in (32, 128):
            raise ValueError('Only explicit 32/128 history contracts are implemented')
        if self.schema_version != 1 or self.inference != 'incremental_v1':
            raise ValueError('Unsupported runtime contract')
        if self.history_protocol != 'zero_state_blocks_retain_previous_action' or self.reward_protocol != 'task_origin_reset':
            raise ValueError('Unsupported history/reset semantics')
        for names in (self.policy_tasks, self.reward_tasks):
            if not names or any(not isinstance(n,str) or not n for n in names) or len(set(names)) != len(names):
                raise ValueError('Unique nonempty task names required')
        if 'general_behavior' in self.reward_tasks or not set(self.reward_tasks) <= set(self.policy_tasks):
            raise ValueError('Reward tasks must be policy skills; general_behavior is BC-only')
        for name in ('policy_state_sha256','reward_state_sha256','feature_contract_sha256','action_spec_sha256'):
            value = getattr(self, name)
            if not isinstance(value,str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
                raise ValueError('Explicit SHA256 required: '+name)

    @property
    def fingerprint(self):
        payload = json.dumps(asdict(self),sort_keys=True,separators=(',',':'))
        return hashlib.sha256(payload.encode()).hexdigest()

    def reward_task(self, policy_task):
        if type(policy_task) is not int or not 0 <= policy_task < len(self.policy_tasks):
            raise ValueError('Unknown policy task')
        name = self.policy_tasks[policy_task]
        if name not in self.reward_tasks:
            raise ValueError('No compatible reward for policy task: '+name)
        return self.reward_tasks.index(name)

    def verify_modules(self, policy, reward):
        for module in (policy,reward):
            if module.training or any(p.requires_grad for p in module.parameters()):
                raise ValueError('Contract requires frozen eval modules')
        if policy.task is None or policy.task.num_embeddings != len(self.policy_tasks):
            raise ValueError('Policy task vocabulary does not match contract')
        if reward.task.num_embeddings != len(self.reward_tasks):
            raise ValueError('Reward task vocabulary does not match contract')
        if reward.transition.in_features != 2*policy.history.hidden_size:
            raise ValueError('Reward uses a different history representation')
        if module_sha256(policy) != self.policy_state_sha256 or module_sha256(reward) != self.reward_state_sha256:
            raise ValueError('Loaded policy/reward state does not match contract')

    def verify_training_metadata(self, policy_metadata, reward_metadata):
        """Require persisted training semantics; names/context cannot be invented at load."""
        if tuple(policy_metadata.get('tasks',())) != self.policy_tasks or policy_metadata.get('context') != self.context:
            raise ValueError('Policy training task order/context mismatch')
        expected = dict(policy_state_sha256=self.policy_state_sha256, context=self.context,
            tasks=list(self.reward_tasks), feature_contract_sha256=self.feature_contract_sha256,
            action_spec_sha256=self.action_spec_sha256, history_protocol=self.history_protocol,
            reward_protocol=self.reward_protocol)
        if any(reward_metadata.get(k) != v for k,v in expected.items()):
            raise ValueError('Reward training metadata is incompatible; old heads cannot be relabelled')

    def verify_root(self, manifest):
        if manifest.get('runtime_contract_sha256') != self.fingerprint:
            raise ValueError('Root runtime contract mismatch')
        if manifest.get('feature_contract_sha256') != self.feature_contract_sha256:
            raise ValueError('Root feature contract mismatch')

    def save(self, path):
        Path(path).write_text(json.dumps(asdict(self),indent=2))

    @classmethod
    def load(cls, path):
        return cls(**json.loads(Path(path).read_text()))


class StreamingReadout:
    """O(T) recurrent readout, same block resets as offline BC, separate reward memory.

    Does not cache world-model outputs or reuse imagined trajectories for updates.
    All state belongs to this instance and is cleared at each task-origin root.
    """
    def __init__(self, policy, reward, contract):
        contract.verify_modules(policy,reward)
        self.policy,self.reward,self.contract = policy,reward,contract
        self.ready = False

    @torch.no_grad()
    def reset(self, feature, past_action, task):
        self.ready = False
        self.task = self.contract.reward_task(task)
        self.policy_state = self.reward_state = None
        h,self.policy_state = self.policy.encode_step(feature[None],past_action[None])
        self.history = h[0].detach().clone()
        self.position = 1
        self.ready = True
        return self.history.clone()

    @torch.no_grad()
    def step(self, feature, past_action):
        if not self.ready:
            raise RuntimeError('Readout reset required')
        self.ready = False
        boundary = self.position == self.contract.context
        if boundary:
            self.policy_state = None
            self.position = 0
        h,self.policy_state = self.policy.encode_step(feature[None],past_action[None],self.policy_state)
        task = torch.tensor([self.task],device=h.device,dtype=torch.long)
        scores,self.reward_state = self.reward.forward_step(self.history[None],h,task,self.reward_state)
        probability = float(scores['completion'][0].sigmoid())
        self.history = h[0].detach().clone()
        self.position += 1
        self.ready = True
        return self.history.clone(),probability,boundary
