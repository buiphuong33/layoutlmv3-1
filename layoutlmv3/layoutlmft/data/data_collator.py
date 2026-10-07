# data_collator.py
import torch
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

from transformers import BatchEncoding, PreTrainedTokenizerBase
from transformers.data.data_collator import (
    DataCollatorMixin,
    _torch_collate_batch,
)
from transformers.file_utils import PaddingStrategy

from typing import NewType
InputDataClass = NewType("InputDataClass", Any)


def pre_calc_rel_mat(segment_ids):
    valid_span = torch.zeros((segment_ids.shape[0], segment_ids.shape[1], segment_ids.shape[1]),
                             device=segment_ids.device, dtype=torch.bool)
    for i in range(segment_ids.shape[0]):
        for j in range(segment_ids.shape[1]):
            valid_span[i, j, :] = segment_ids[i, :] == segment_ids[i, j]

    return valid_span


@dataclass
class DataCollatorForKeyValueExtraction(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.
    
    ★ Hỗ trợ boundary labels cho Boundary Loss:
       - use_boundary=True: tạo boundary_labels (binary) cho mỗi token
       - boundary_label_ids: list các label ID được coi là "B-" (bắt đầu entity)
       - Nếu boundary_label_ids=None → dùng heuristic transition-based
    """
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    # ★ Boundary loss params
    use_boundary: bool = False
    boundary_label_ids: Optional[List[int]] = None

    def __call__(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None

        images = None
        if "images" in features[0]:
            images = torch.stack([torch.tensor(d.pop("images")) for d in features])
            IMAGE_LEN = int(images.shape[-1] / 16) * int(images.shape[-1] / 16) + 1

        batch = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="pt" if labels is None else None,
        )

        if images is not None:
            batch["images"] = images
            batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) and k == 'attention_mask' else v
                     for k, v in batch.items()}
            visual_attention_mask = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long)
            batch["attention_mask"] = torch.cat([batch['attention_mask'], visual_attention_mask], dim=1)

        if labels is None:
            return batch

        has_bbox_input = "bbox" in features[0]
        has_position_input = "position_ids" in features[0]
        padding_idx = self.tokenizer.pad_token_id
        sequence_length = torch.tensor(batch["input_ids"]).shape[1]
        padding_side = self.tokenizer.padding_side

        if padding_side == "right":
            batch["labels"] = [label + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels]
            if has_bbox_input:
                batch["bbox"] = [bbox + [[0, 0, 0, 0]] * (sequence_length - len(bbox)) for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [position_id + [padding_idx] * (sequence_length - len(position_id))
                                          for position_id in batch["position_ids"]]
        else:
            batch["labels"] = [[self.label_pad_token_id] * (sequence_length - len(label)) + label for label in labels]
            if has_bbox_input:
                batch["bbox"] = [[[0, 0, 0, 0]] * (sequence_length - len(bbox)) + bbox for bbox in batch["bbox"]]
            if has_position_input:
                batch["position_ids"] = [[padding_idx] * (sequence_length - len(position_id))
                                          + position_id for position_id in batch["position_ids"]]

        # ★ BOUNDARY LABELS
        if self.use_boundary:
            boundary_labels = []
            for label_seq in labels:
                bl = []
                for i, l in enumerate(label_seq):
                    # Padding / special token → ignore
                    if l == self.label_pad_token_id:
                        bl.append(self.label_pad_token_id)
                    # Nếu có danh sách B-* IDs → dùng trực tiếp
                    elif self.boundary_label_ids is not None:
                        bl.append(1 if l in self.boundary_label_ids else 0)
                    # Fallback: transition-based
                    else:
                        if i == 0:
                            bl.append(1 if l > 0 else 0)
                        else:
                            prev_l = label_seq[i - 1]
                            if prev_l == self.label_pad_token_id:
                                bl.append(0)
                            else:
                                bl.append(1 if l != prev_l else 0)
                boundary_labels.append(bl)

            # Pad boundary_labels (giống labels)
            if padding_side == "right":
                batch["boundary_labels"] = [
                    bl + [self.label_pad_token_id] * (sequence_length - len(bl))
                    for bl in boundary_labels
                ]
            else:
                batch["boundary_labels"] = [
                    [self.label_pad_token_id] * (sequence_length - len(bl)) + bl
                    for bl in boundary_labels
                ]

        if 'segment_ids' in batch:
            assert 'position_ids' in batch
            for i in range(len(batch['segment_ids'])):
                batch['segment_ids'][i] = batch['segment_ids'][i] + [batch['segment_ids'][i][-1] + 1] * (sequence_length - len(batch['segment_ids'][i])) + [
                    batch['segment_ids'][i][-1] + 2] * IMAGE_LEN

        batch = {k: torch.tensor(v, dtype=torch.int64) if isinstance(v[0], list) else v for k, v in batch.items()}

        if 'segment_ids' in batch:
            valid_span = pre_calc_rel_mat(
                segment_ids=batch['segment_ids']
            )
            batch['valid_span'] = valid_span
            del batch['segment_ids']

        if images is not None:
            visual_labels = torch.ones((len(batch['input_ids']), IMAGE_LEN), dtype=torch.long) * -100
            batch["labels"] = torch.cat([batch['labels'], visual_labels], dim=1)
            
            # ★ Boundary labels cho visual tokens: ignore (label = -100)
            if self.use_boundary:
                visual_boundary_labels = torch.ones(
                    (len(batch['input_ids']), IMAGE_LEN), dtype=torch.long
                ) * -100
                batch["boundary_labels"] = torch.cat(
                    [batch['boundary_labels'], visual_boundary_labels], dim=1
                )

        return batch