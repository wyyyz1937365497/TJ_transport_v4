"""
Imitation Learning Dataset for OCR-MAX

This module provides dataset classes for behavioral cloning,
supporting both expert demonstrations and MPC-generated demonstrations.

Key Features:
1. Efficient loading of demonstrations from pickle files
2. Proper handling of variable-length vehicle observations
3. Support for action masking (vehicles not under control)
4. Train/validation split with proper stratification
5. Data quality filtering (removes low-OCR episodes)
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json


class ImitationDataset(Dataset):
    """
    Imitation Learning Dataset for Behavioral Cloning

    This dataset loads expert demonstrations and provides
    (observation, action) pairs for training the policy network
    through behavioral cloning.

    Data Format:
        - Observations: [num_vehicles * 9 + 32 + 1] = 321 dims
          * num_ehicles * 9: Vehicle states (s, d, v, a, lane, etc.)
          * + 32: Global statistics
          * + 1: Actual number of vehicles
        - Actions: [num_vehicles * 2] flattened
          * Each vehicle: [acceleration, lane_change_probability]
          * acceleration: [-3, 2] m/s²
          * lane_change: [0, 1] probability

    Args:
        demonstrations_path: Path to pickle file containing demonstrations
        filter_low_ocr: Whether to filter episodes with OCR < baseline (0.5469)
        min_ocr_threshold: Minimum OCR threshold for filtering
        transform: Optional transform to apply to observations
        action_transform: Optional transform to apply to actions
    """

    def __init__(
        self,
        demonstrations_path: str,
        filter_low_ocr: bool = True,
        min_ocr_threshold: float = 0.5469,
        transform=None,
        action_transform=None
    ):
        self.demonstrations_path = Path(demonstrations_path)
        self.filter_low_ocr = filter_low_ocr
        self.min_ocr_threshold = min_ocr_threshold
        self.transform = transform
        self.action_transform = action_transform

        # Load demonstrations
        print(f"[ImitationDataset] Loading demonstrations from {demonstrations_path}")
        with open(self.demonstrations_path, 'rb') as f:
            demo_data = pickle.load(f)

        # Handle different data formats
        if isinstance(demo_data, dict) and 'episodes' in demo_data:
            # Format: {'episodes': [...], 'metadata': {...}}
            episodes = demo_data['episodes']
            self.metadata = demo_data.get('metadata', {})
        elif isinstance(demo_data, list):
            # Format: [episode1, episode2, ...]
            episodes = demo_data
            self.metadata = {}
        else:
            raise ValueError(f"Unsupported demonstration format: {type(demo_data)}")

        print(f"[ImitationDataset] Loaded {len(episodes)} episodes")

        # Filter episodes by OCR if enabled
        if self.filter_low_ocr:
            filtered_episodes = []
            for ep in episodes:
                ocr = ep.get('ocr', ep.get('final_ocr', 0.0))
                if ocr >= self.min_ocr_threshold:
                    filtered_episodes.append(ep)
                else:
                    print(f"[ImitationDataset] Filtered out episode with OCR={ocr:.4f} < {min_ocr_threshold}")

            self.episodes = filtered_episodes
            print(f"[ImitationDataset] After filtering: {len(self.episodes)} episodes")
        else:
            self.episodes = episodes

        # Flatten transitions
        self.transitions = []
        self._flatten_transitions()

        print(f"[ImitationDataset] Total transitions: {len(self.transitions)}")
        print(f"[ImitationDataset] Dataset initialization complete")

    def _flatten_transitions(self):
        """Flatten episodes into individual transitions"""
        for ep_idx, episode in enumerate(self.episodes):
            ocr = episode.get('ocr', episode.get('final_ocr', 0.0))
            transitions = episode.get('transitions', [])

            for trans_idx, trans in enumerate(transitions):
                # Validate transition
                if 'obs' not in trans or 'actions' not in trans:
                    continue

                # Only add transitions with valid actions
                if trans['actions'] is not None and len(trans['actions']) > 0:
                    self.transitions.append({
                        'episode_idx': ep_idx,
                        'transition_idx': trans_idx,
                        'ocr': ocr,
                        'obs': trans['obs'],
                        'actions': trans['actions'],
                        'icv_ids': trans.get('icv_ids', []),
                        'vehicle_ids': trans.get('vehicle_ids', []),
                        'mask': trans.get('mask', None)
                    })

    def __len__(self) -> int:
        return len(self.transitions)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single transition

        Returns:
            {
                'obs': [321] observation tensor,
                'actions': [64] flattened action tensor (32 vehicles * 2),
                'mask': [32] boolean mask (1=controlled, 0=not controlled),
                'episode_idx': int,
                'ocr': float
            }
        """
        trans = self.transitions[idx]

        # Extract observation
        obs = np.array(trans['obs'], dtype=np.float32)

        # Extract and flatten actions
        actions_dict = trans['actions']
        icv_ids = trans.get('icv_ids', [])
        vehicle_ids = trans.get('vehicle_ids', [])

        # Create flattened action array [32 * 2] = [64]
        # Default: no action (acceleration=0, lane_change=0)
        flat_actions = np.zeros(32 * 2, dtype=np.float32)

        # Create mask: which vehicles are controlled
        mask = np.zeros(32, dtype=np.float32)

        # Fill in actions for ICV vehicles
        for i, veh_id in enumerate(icv_ids[:32]):  # Max 32 vehicles
            if veh_id in actions_dict:
                action = actions_dict[veh_id]
                # action should be [acceleration, lane_change_probability]
                if isinstance(action, (list, tuple, np.ndarray)) and len(action) >= 2:
                    flat_actions[i * 2] = float(action[0])      # acceleration
                    flat_actions[i * 2 + 1] = float(action[1])  # lane_change
                    mask[i] = 1.0

        # Apply transforms if provided
        if self.transform is not None:
            obs = self.transform(obs)

        if self.action_transform is not None:
            flat_actions = self.action_transform(flat_actions)

        return {
            'obs': torch.from_numpy(obs),
            'actions': torch.from_numpy(flat_actions),
            'mask': torch.from_numpy(mask),
            'episode_idx': trans['episode_idx'],
            'ocr': trans['ocr']
        }


class MPCImitationDataset(ImitationDataset):
    """
    Specialized dataset for MPC-generated demonstrations

    MPC demonstrations have additional information:
    - Solving time statistics
    - Optimization success/failure flags
    - Constraint violations

    This dataset can filter based on MPC solving success
    and provide additional quality metrics.
    """

    def __init__(
        self,
        demonstrations_path: str,
        filter_low_ocr: bool = True,
        min_ocr_threshold: float = 0.5469,
        min_mpc_success_rate: float = 0.8,  # Only use episodes with >80% successful MPC solves
        transform=None,
        action_transform=None
    ):
        self.min_mpc_success_rate = min_mpc_success_rate
        super().__init__(
            demonstrations_path,
            filter_low_ocr,
            min_ocr_threshold,
            transform,
            action_transform
        )

    def _flatten_transitions(self):
        """Flatten episodes with MPC quality filtering"""
        for ep_idx, episode in enumerate(self.episodes):
            ocr = episode.get('ocr', episode.get('final_ocr', 0.0))
            transitions = episode.get('transitions', [])

            # Calculate MPC success rate for this episode
            mpc_successes = sum(
                1 for t in transitions
                if t.get('mpc_solve_success', True)
            )
            mpc_success_rate = mpc_successes / len(transitions) if transitions else 0.0

            # Filter episodes with low MPC success rate
            if mpc_success_rate < self.min_mpc_success_rate:
                print(f"[MPCImitationDataset] Filtered episode {ep_idx} with "
                      f"MPC success rate {mpc_success_rate:.2%}")
                continue

            for trans_idx, trans in enumerate(transitions):
                # Validate transition
                if 'obs' not in trans or 'mpc_action' not in trans:
                    continue

                # Only add successful MPC solves
                if not trans.get('mpc_solve_success', True):
                    continue

                # Extract MPC action
                mpc_action = trans['mpc_action']
                if mpc_action is None:
                    continue

                # Convert MPC action to action dict format
                # MPC action is typically [acceleration, lane_change] for all vehicles
                actions_dict = self._mpc_action_to_dict(mpc_action, trans)

                self.transitions.append({
                    'episode_idx': ep_idx,
                    'transition_idx': trans_idx,
                    'ocr': ocr,
                    'obs': trans['obs'],
                    'actions': actions_dict,
                    'icv_ids': trans.get('icv_ids', []),
                    'vehicle_ids': trans.get('vehicle_ids', []),
                    'mask': trans.get('mask', None),
                    'mpc_solve_time': trans.get('mpc_solve_time', 0.0)
                })

    def _mpc_action_to_dict(
        self,
        mpc_action: np.ndarray,
        trans: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """Convert MPC action array to action dictionary format"""
        actions_dict = {}
        icv_ids = trans.get('icv_ids', [])

        # mpc_action shape: [32, 2] or [64] (flattened)
        if len(mpc_action.shape) == 1:
            # Flattened: [64] -> reshape to [32, 2]
            mpc_action = mpc_action.reshape(32, 2)

        for i, veh_id in enumerate(icv_ids[:32]):
            if i < mpc_action.shape[0]:
                actions_dict[veh_id] = mpc_action[i]

        return actions_dict


def create_dataloaders(
    demonstrations_path: str,
    batch_size: int = 32,
    val_split: float = 0.2,
    num_workers: int = 4,
    filter_low_ocr: bool = True,
    min_ocr_threshold: float = 0.5469,
    dataset_type: str = 'expert',
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, Dict[str, Any]]:
    """
    Create train and validation dataloaders

    Args:
        demonstrations_path: Path to demonstrations pickle file
        batch_size: Batch size for training
        val_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        filter_low_ocr: Whether to filter low OCR episodes
        min_ocr_threshold: Minimum OCR threshold
        dataset_type: 'expert' or 'mpc'
        seed: Random seed for splitting

    Returns:
        train_loader, val_loader, metadata
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Create dataset
    if dataset_type == 'mpc':
        dataset = MPCImitationDataset(
            demonstrations_path,
            filter_low_ocr=filter_low_ocr,
            min_ocr_threshold=min_ocr_threshold
        )
    else:
        dataset = ImitationDataset(
            demonstrations_path,
            filter_low_ocr=filter_low_ocr,
            min_ocr_threshold=min_ocr_threshold
        )

    # Split into train and validation
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(seed)
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Compute metadata
    metadata = {
        'total_transitions': len(dataset),
        'train_transitions': train_size,
        'val_transitions': val_size,
        'num_train_batches': len(train_loader),
        'num_val_batches': len(val_loader),
        'dataset_metadata': dataset.metadata
    }

    return train_loader, val_loader, metadata


if __name__ == '__main__':
    """Test the dataset loader"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--demo_path', type=str, required=True,
                        help='Path to demonstrations pickle file')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    args = parser.parse_args()

    # Create dataloaders
    train_loader, val_loader, metadata = create_dataloaders(
        demonstrations_path=args.demo_path,
        batch_size=args.batch_size,
        val_split=0.2,
        num_workers=args.num_workers,
        filter_low_ocr=True,
        min_ocr_threshold=0.5469
    )

    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")
    print(f"Total transitions: {metadata['total_transitions']}")
    print(f"Train transitions: {metadata['train_transitions']}")
    print(f"Val transitions: {metadata['val_transitions']}")
    print(f"Train batches: {metadata['num_train_batches']}")
    print(f"Val batches: {metadata['num_val_batches']}")

    # Test loading a batch
    print(f"\n{'='*60}")
    print("Testing Batch Loading")
    print(f"{'='*60}")

    for batch in train_loader:
        print(f"Batch shapes:")
        print(f"  - obs: {batch['obs'].shape}")
        print(f"  - actions: {batch['actions'].shape}")
        print(f"  - mask: {batch['mask'].shape}")
        print(f"  - episode_idx: {batch['episode_idx'].shape}")
        print(f"  - ocr: {batch['ocr'].shape}")

        print(f"\nBatch statistics:")
        print(f"  - obs range: [{batch['obs'].min():.3f}, {batch['obs'].max():.3f}]")
        print(f"  - actions range: [{batch['actions'].min():.3f}, {batch['actions'].max():.3f}]")
        print(f"  - controlled vehicles: {batch['mask'].sum().item()}/{batch['mask'].numel()}")
        print(f"  - avg OCR: {batch['ocr'].mean().item():.4f}")

        break  # Only test first batch
