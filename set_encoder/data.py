import random
from pathlib import Path
from typing import Literal

import pandas as pd
import torch
from lightning import Callback
from lightning_ir.data.dataset import RankSample, RunDataset, TupleDataset
from lightning_ir.data.external_datasets import register_new_dataset


def register_rank_distillm_novelty():

    register_new_dataset(
        "task2-collie-2025",
        docs=Path(__file__).parent.parent / "data" / "task2-collie-2025-docs.tsv",
        queries=Path(__file__).parent.parent / "data" / "task2-collie-2025-queries.tsv",
        qrels=Path(__file__).parent.parent / "data" / "task2-collie-2025-qrels.qrels",
        scoreddocs=Path(__file__).parent.parent / "data" / "task2-collie-2025-scoreddocs.tsv",
    )


class RegisterRankDistiLLMNovelty(Callback):
    def __init__(self) -> None:
        super().__init__()
        register_rank_distillm_novelty()


class SubtopicRunDataset(RunDataset):

    def __init__(
        self,
        run_path_or_id: Path | str,
        depth: int = -1,
        sample_size: int = -1,
        sampling_strategy: Literal["single_relevant", "top", "random", "log_random", "top_and_random"] = "top",
        targets: Literal["relevance", "subtopic_relevance", "rank", "score"] | None = None,
        normalize_targets: bool = False,
        add_docs_not_in_ranking: bool = False,
    ) -> None:
        super().__init__(
            run_path_or_id=run_path_or_id,
            depth=depth,
            sample_size=sample_size,
            sampling_strategy=sampling_strategy,
            targets=targets,
            normalize_targets=normalize_targets,
            add_docs_not_in_ranking=add_docs_not_in_ranking,
        )
        self.targets = "sub_topics"

    def load_qrels(self, *args, **kwargs) -> pd.DataFrame | None:
        return None

    def __getitem__(self, idx: int) -> RankSample:
        query_id = str(self.query_ids[idx])
        group = self.run_groups.get_group(query_id).copy()
        query = self.queries[query_id]
        group = group.head(self.sample_size)

        doc_ids = tuple(group["doc_id"])
        docs = tuple(self.docs.get(doc_id).default_text() for doc_id in doc_ids)

        targets = torch.tensor(group["iteration"].values)
        return RankSample(query_id, query, doc_ids, docs, targets)


class RepeatRunDataset(RunDataset):

    def __getitem__(self, idx: int) -> RankSample:
        sample = super().__getitem__(idx)
        doc_ids = sample.doc_ids
        docs = sample.docs
        repeat_idx = random.randint(0, len(doc_ids) - 1)
        doc_ids = doc_ids + (doc_ids[repeat_idx],)
        docs = docs + (docs[repeat_idx],)
        targets = None
        if sample.targets is not None:
            repeat_targets = torch.zeros((len(doc_ids), 1))
            repeat_targets[[repeat_idx, -1]] = 1
            original_targets = sample.targets
            original_targets = torch.cat([original_targets, torch.zeros((1, 1))])
            targets = torch.cat([original_targets, repeat_targets], axis=1)
        sample = RankSample(sample.query_id, sample.query, doc_ids, docs, targets)
        return sample