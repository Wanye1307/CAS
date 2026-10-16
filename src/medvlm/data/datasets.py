"""
Dataset loading and preprocessing for Medical VQA.

Provides a unified interface for different VQA datasets with support for:
- Question type filtering (closed/open/all)
- Subsampling with seed control
- Conversion to unified format
- Local file loading (for datasets like SLAKE)
- Optimized image handling with HuggingFace Image feature
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Union
from datasets import Dataset, Features, Image, Value, load_dataset
import json
import os

from ..configs import DataConfig, QuestionType, DatasetName


# Unified schema for all VQA datasets
VQA_FEATURES = Features({
    "image": Image(),  # HuggingFace optimized image handling
    "question": Value("string"),
    "answer": Value("string"),
    "answer_type": Value("string"),  # "closed" or "open"
    "question_id": Value("string"),
    "image_id": Value("string"),
    "dataset_source": Value("string"),
})


@dataclass
class VQASample:
    """Unified VQA sample format across all datasets."""
    image: Any  # PIL Image or path
    question: str
    answer: str
    answer_type: str  # "closed" or "open"

    # Optional metadata
    question_id: Optional[str] = None
    image_id: Optional[str] = None
    dataset_source: Optional[str] = None
    extra_metadata: Optional[Dict] = None


class BaseVQADataset(ABC):
    """Abstract base class for VQA datasets."""

    def __init__(self, config: DataConfig):
        self.config = config
        self._raw_dataset: Optional[Dataset] = None
        self._processed_dataset: Optional[Dataset] = None

    @property
    @abstractmethod
    def name(self) -> str:
        """Dataset name identifier."""
        pass

    @abstractmethod
    def _load_raw(self) -> Dataset:
        """Load raw dataset and return in unified format with VQA_FEATURES."""
        pass

    def load(self) -> Dataset:
        """Load and process the dataset."""
        # Load raw data (already in unified format)
        self._processed_dataset = self._load_raw()

        # Filter by question type
        self._processed_dataset = self._filter_by_question_type()

        # Shuffle with seed
        self._processed_dataset = self._processed_dataset.shuffle(seed=self.config.seed)

        # Subsample if requested
        if self.config.subsample_size is not None:
            self._processed_dataset = self._subsample()

        return self._processed_dataset

    def _filter_by_question_type(self) -> Dataset:
        """Filter dataset by question type."""
        if self.config.question_type == QuestionType.ALL:
            return self._processed_dataset

        target_type = self.config.question_type.value

        original_size = len(self._processed_dataset)
        filtered = self._processed_dataset.filter(
            lambda x: x["answer_type"] == target_type
        )

        print(f"[{self.name}] Filtered {self.config.question_type.value} questions: "
              f"{len(filtered)}/{original_size} ({len(filtered)/original_size*100:.1f}%)")

        return filtered

    def _subsample(self) -> Dataset:
        """Subsample dataset with seed control."""
        target_size = min(self.config.subsample_size, len(self._processed_dataset))

        # Use deterministic selection after shuffle
        subsampled = self._processed_dataset.select(range(target_size))

        print(f"[{self.name}] Subsampled to {target_size} examples (seed={self.config.seed})")

        return subsampled

    def get_statistics(self) -> Dict[str, Any]:
        """Get dataset statistics (counts, ratios, filter/subsample settings).

        Returns:
            Dict with keys: name, total_samples, closed_questions,
            open_questions, closed_ratio, question_type_filter,
            subsample_size, seed.
        """
        if self._processed_dataset is None:
            raise ValueError("Dataset not loaded. Call load() first.")

        closed_count = sum(1 for x in self._processed_dataset if x["answer_type"] == "closed")
        open_count = len(self._processed_dataset) - closed_count

        return {
            "name": self.name,
            "total_samples": len(self._processed_dataset),
            "closed_questions": closed_count,
            "open_questions": open_count,
            "closed_ratio": closed_count / len(self._processed_dataset) if len(self._processed_dataset) > 0 else 0,
            "question_type_filter": self.config.question_type.value,
            "subsample_size": self.config.subsample_size,
            "seed": self.config.seed,
        }


class VQARADDataset(BaseVQADataset):
    """VQA-RAD (Radiology VQA) dataset."""

    HF_DATASET_ID = "flaviagiammarino/vqa-rad"

    @property
    def name(self) -> str:
        return "VQA-RAD"

    def _determine_answer_type(self, answer: str) -> str:
        """VQA-RAD: closed if answer is yes/no."""
        answer_lower = answer.lower().strip()
        return "closed" if answer_lower in ["yes", "no"] else "open"

    def _load_raw(self) -> Dataset:
        """Load VQA-RAD from HuggingFace and convert to unified format."""
        print(f"[{self.name}] Loading from HuggingFace: {self.HF_DATASET_ID}")
        raw_dataset = load_dataset(self.HF_DATASET_ID, split=self.config.split)

        print(f"[{self.name}] Converting {len(raw_dataset)} samples to unified format...")

        # Build unified samples list
        samples = []
        for idx, sample in enumerate(raw_dataset):
            answer = str(sample["answer"])
            samples.append({
                "image": sample["image"],  # Already PIL Image from HF
                "question": sample["question"],
                "answer": answer,
                "answer_type": self._determine_answer_type(answer),
                "question_id": f"vqa_rad_q_{idx}",
                "image_id": f"vqa_rad_img_{idx}",
                "dataset_source": self.name,
            })

        # Create dataset with optimized features
        dataset = Dataset.from_list(samples, features=VQA_FEATURES)
        print(f"[{self.name}] Loaded {len(dataset)} samples")

        return dataset


class SLAKEDataset(BaseVQADataset):
    """SLAKE (Semantically-Labeled Knowledge-Enhanced) Medical VQA dataset.

    Loads from HuggingFace (BoKelvin/SLAKE) by default. Falls back to local
    files if data_path is specified in DataConfig.

    HuggingFace dataset: https://huggingface.co/datasets/BoKelvin/SLAKE
    Only English questions are used. CLOSED yes/no questions stay as "closed";
    other CLOSED answers (e.g., multiple choice) are treated as "open".

    For local loading, set data_path in DataConfig:
        DataConfig(
            dataset_name=DatasetName.SLAKE,
            data_path="/path/to/Slake1.0",
            ...
        )
    """

    HF_DATASET_ID = "BoKelvin/SLAKE"

    # Default local paths to check (used only when data_path is set)
    DEFAULT_LOCAL_PATHS = [
        "./data/Slake1.0",
    ]

    # HuggingFace split names differ from local file names
    HF_SPLIT_MAPPING = {
        "train": "train",
        "test": "test",
        "validation": "validation",
        "validate": "validation",
    }

    @property
    def name(self) -> str:
        return "SLAKE"

    def _normalize_answer_type(self, answer_type: str, answer: str) -> str:
        """Normalize SLAKE answer_type.

        CLOSED with yes/no answer -> "closed"
        CLOSED with other answer  -> "open"
        OPEN                      -> "open"
        """
        answer_type_upper = str(answer_type).upper()
        if answer_type_upper == "CLOSED" and answer.lower().strip() in ("yes", "no"):
            return "closed"
        return "open"

    def _load_from_huggingface(self) -> Dataset:
        """Load SLAKE from HuggingFace Hub, filtering to English only."""
        hf_split = self.HF_SPLIT_MAPPING.get(self.config.split)
        if hf_split is None:
            raise ValueError(f"Unknown split: {self.config.split}. "
                           f"Available: {list(self.HF_SPLIT_MAPPING.keys())}")

        print(f"[{self.name}] Loading from HuggingFace: {self.HF_DATASET_ID} (split={hf_split})")
        raw_dataset = load_dataset(self.HF_DATASET_ID, split=hf_split)

        # Filter to English only
        english_dataset = raw_dataset.filter(lambda x: x.get("q_lang") == "en")
        print(f"[{self.name}] Filtered to English: {len(english_dataset)}/{len(raw_dataset)}")

        # Convert to unified format
        samples = []
        for idx, sample in enumerate(english_dataset):
            answer = str(sample["answer"])
            qid = sample.get("qid", idx)
            img_id = sample.get("img_id", idx)

            samples.append({
                "image": sample["image"],  # PIL Image from HF
                "question": sample["question"],
                "answer": answer,
                "answer_type": self._normalize_answer_type(
                    sample.get("answer_type", "OPEN"), answer
                ),
                "question_id": f"slake_q_{qid}",
                "image_id": f"slake_img_{img_id}",
                "dataset_source": self.name,
            })

        dataset = Dataset.from_list(samples, features=VQA_FEATURES)
        print(f"[{self.name}] Loaded {len(dataset)} samples")
        return dataset

    def _find_local_path(self) -> Optional[str]:
        """Find SLAKE dataset in local locations. Returns None if not found."""
        if self.config.data_path:
            expanded = os.path.expanduser(self.config.data_path)
            if os.path.exists(expanded):
                return expanded
            raise FileNotFoundError(f"SLAKE data_path not found: {self.config.data_path}")

        # Check default locations
        for path in self.DEFAULT_LOCAL_PATHS:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                print(f"[{self.name}] Found local dataset at: {expanded}")
                return expanded

        return None

    def _load_from_local(self, data_path: str) -> Dataset:
        """Load SLAKE from local files with optimized image handling."""
        # Map split names
        split_mapping = {
            "train": "train.json",
            "test": "test.json",
            "validation": "validate.json",
            "validate": "validate.json",
        }

        split_file = split_mapping.get(self.config.split)
        if split_file is None:
            raise ValueError(f"Unknown split: {self.config.split}. "
                           f"Available: {list(split_mapping.keys())}")

        json_path = os.path.join(data_path, split_file)
        imgs_dir = os.path.join(data_path, "imgs")

        if not os.path.exists(json_path):
            raise FileNotFoundError(f"Split file not found: {json_path}")
        if not os.path.exists(imgs_dir):
            raise FileNotFoundError(f"Images directory not found: {imgs_dir}")

        print(f"[{self.name}] Loading from local: {json_path}")

        # Load JSON annotations
        with open(json_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        # Filter to English only
        english_annotations = [a for a in annotations if a.get("q_lang") == "en"]
        print(f"[{self.name}] Filtered to English: {len(english_annotations)}/{len(annotations)}")

        # Build samples with image PATHS (not loaded PIL images)
        samples = []
        missing_images = 0

        for idx, ann in enumerate(english_annotations):
            img_name = ann.get("img_name", "")
            img_path = os.path.join(imgs_dir, img_name)

            if os.path.exists(img_path):
                qid = ann.get("qid", idx)
                img_id = ann.get("img_id", idx)

                samples.append({
                    "image": img_path,
                    "question": ann["question"],
                    "answer": str(ann["answer"]),
                    "answer_type": self._normalize_answer_type(
                        ann.get("answer_type", "OPEN"), str(ann["answer"])
                    ),
                    "question_id": f"slake_q_{qid}",
                    "image_id": f"slake_img_{img_id}",
                    "dataset_source": self.name,
                })
            else:
                missing_images += 1

        if missing_images > 0:
            print(f"[{self.name}] Warning: {missing_images} images not found")

        dataset = Dataset.from_list(samples, features=VQA_FEATURES)
        print(f"[{self.name}] Loaded {len(dataset)} samples (images loaded lazily)")
        return dataset

    def _load_raw(self) -> Dataset:
        """Load SLAKE dataset. Uses local files if data_path is set,
        otherwise downloads from HuggingFace."""
        # If data_path is explicitly set, use local loading
        if self.config.data_path:
            local_path = self._find_local_path()
            return self._load_from_local(local_path)

        # Check default local paths first (avoids re-downloading)
        for path in self.DEFAULT_LOCAL_PATHS:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                print(f"[{self.name}] Found local dataset at: {expanded}")
                return self._load_from_local(expanded)

        # Fall back to HuggingFace
        return self._load_from_huggingface()


class VQAMed2019Dataset(BaseVQADataset):
    """VQA-Med-2019 dataset.

    500 questions across 4 categories: modality, plane, organ, abnormality.
    Mix of yes/no and short-answer questions.

    Format: image_id|category|question|answer (pipe-separated, 4 fields)
    Images: VQAMed2019_Test_Images/{image_id}.jpg
    """

    DEFAULT_LOCAL_PATHS = [
        "./data/vqa_med_2019/VQAMed2019Test",
    ]

    @property
    def name(self) -> str:
        return "VQA-Med-2019"

    def _find_local_path(self) -> str:
        if self.config.data_path:
            expanded = os.path.expanduser(self.config.data_path)
            if os.path.exists(expanded):
                return expanded
            raise FileNotFoundError(f"VQA-Med-2019 data_path not found: {self.config.data_path}")
        for path in self.DEFAULT_LOCAL_PATHS:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded
        raise FileNotFoundError(
            f"VQA-Med-2019 dataset not found. Please set data_path in DataConfig.\n"
            f"Checked locations:\n" +
            "\n".join(f"  - {p}" for p in self.DEFAULT_LOCAL_PATHS)
        )

    def _determine_answer_type(self, answer: str, category: str) -> str:
        """Determine answer type based on answer content and category."""
        answer_lower = answer.lower().strip()
        if answer_lower in ("yes", "no"):
            return "closed"
        return "open"

    def _load_raw(self) -> Dataset:
        data_path = self._find_local_path()
        answers_file = os.path.join(data_path, "VQAMed2019_Test_Questions_w_Ref_Answers.txt")
        imgs_dir = os.path.join(data_path, "VQAMed2019_Test_Images")

        if not os.path.exists(answers_file):
            raise FileNotFoundError(f"Answers file not found: {answers_file}")
        if not os.path.exists(imgs_dir):
            raise FileNotFoundError(f"Images directory not found: {imgs_dir}")

        print(f"[{self.name}] Loading from local: {data_path}")

        samples = []
        missing_images = 0

        with open(answers_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) < 4:
                    continue
                image_id = parts[0].strip()
                category = parts[1].strip()
                question = parts[2].strip()
                answer = parts[3].strip()

                img_path = os.path.join(imgs_dir, f"{image_id}.jpg")
                if not os.path.exists(img_path):
                    missing_images += 1
                    continue

                samples.append({
                    "image": img_path,
                    "question": question,
                    "answer": answer,
                    "answer_type": self._determine_answer_type(answer, category),
                    "question_id": f"vqamed2019_q_{image_id}_{category}",
                    "image_id": image_id,
                    "dataset_source": self.name,
                })

        if missing_images > 0:
            print(f"[{self.name}] Warning: {missing_images} images not found")

        dataset = Dataset.from_list(samples, features=VQA_FEATURES)
        print(f"[{self.name}] Loaded {len(dataset)} samples (images loaded lazily)")
        return dataset


class VQAMed2020Dataset(BaseVQADataset):
    """VQA-Med-2020 Task 1 dataset.

    500 abnormality questions. Mix of yes/no (32) and open-ended diagnoses (468).

    Format: image_id|question and image_id|answer (separate files, pipe-separated)
    Images: Task1-2020-VQAnswering-Test-Images/{image_id}.jpg
    """

    DEFAULT_LOCAL_PATHS = [
        "./data/vqa_med_2020/VQA-TestSet-ReferenceAnswers-VQAMed2020-Task1",
    ]

    @property
    def name(self) -> str:
        return "VQA-Med-2020"

    def _find_local_path(self) -> str:
        if self.config.data_path:
            expanded = os.path.expanduser(self.config.data_path)
            if os.path.exists(expanded):
                return expanded
            raise FileNotFoundError(f"VQA-Med-2020 data_path not found: {self.config.data_path}")
        for path in self.DEFAULT_LOCAL_PATHS:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                return expanded
        raise FileNotFoundError(
            f"VQA-Med-2020 dataset not found. Please set data_path in DataConfig.\n"
            f"Checked locations:\n" +
            "\n".join(f"  - {p}" for p in self.DEFAULT_LOCAL_PATHS)
        )

    def _determine_answer_type(self, answer: str) -> str:
        answer_lower = answer.lower().strip()
        if answer_lower in ("yes", "no"):
            return "closed"
        return "open"

    def _load_raw(self) -> Dataset:
        data_path = self._find_local_path()
        questions_file = os.path.join(data_path, "VQAMed2020-Task1-VQAnswering-Test-Questions.txt")
        answers_file = os.path.join(data_path, "VQAMed2020-Task1-VQAnswering-Test-ReferenceAnswers.txt")
        imgs_dir = os.path.join(data_path, "Task1-2020-VQAnswering-Test-Images")

        if not os.path.exists(questions_file):
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        if not os.path.exists(answers_file):
            raise FileNotFoundError(f"Answers file not found: {answers_file}")
        if not os.path.exists(imgs_dir):
            raise FileNotFoundError(f"Images directory not found: {imgs_dir}")

        print(f"[{self.name}] Loading from local: {data_path}")

        questions = {}
        with open(questions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    questions[parts[0].strip()] = parts[1].strip()

        answers = {}
        with open(answers_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    answers[parts[0].strip()] = parts[1].strip()

        samples = []
        missing_images = 0

        for image_id, question in questions.items():
            img_path = os.path.join(imgs_dir, f"{image_id}.jpg")
            if not os.path.exists(img_path):
                missing_images += 1
                continue

            answer = answers.get(image_id, "")
            if not answer:
                continue

            samples.append({
                "image": img_path,
                "question": question,
                "answer": answer,
                "answer_type": self._determine_answer_type(answer),
                "question_id": f"vqamed2020_q_{image_id}",
                "image_id": image_id,
                "dataset_source": self.name,
            })

        if missing_images > 0:
            print(f"[{self.name}] Warning: {missing_images} images not found")

        dataset = Dataset.from_list(samples, features=VQA_FEATURES)
        print(f"[{self.name}] Loaded {len(dataset)} samples (images loaded lazily)")
        return dataset


class VQAMed2021Dataset(BaseVQADataset):
    """VQA-Med-2021 Task 1 (Abnormality) dataset.

    500 open-ended abnormality questions about radiology images.

    Expected directory structure:
        Task1-VQA-2021-TestSet-w-GroundTruth/
        ├── Task1-VQA-2021-TestSet-Questions.txt
        ├── Task1-VQA-2021-TestSet-ReferenceAnswers.txt
        └── images/
            └── VQA-500-Images/
                ├── synpic42072.jpg
                └── ...

    Questions file: image_id|question (pipe-separated)
    Answers file: image_id|answer1|answer2|... (pipe-separated, multiple acceptable)
    """

    DEFAULT_LOCAL_PATHS = [
        "./data/vqa_med_2021/Task1-VQA-2021-TestSet-w-GroundTruth",
    ]

    @property
    def name(self) -> str:
        return "VQA-Med-2021"

    def _find_local_path(self) -> str:
        """Find VQA-Med-2021 dataset in common locations."""
        if self.config.data_path:
            expanded = os.path.expanduser(self.config.data_path)
            if os.path.exists(expanded):
                return expanded
            raise FileNotFoundError(f"VQA-Med-2021 data_path not found: {self.config.data_path}")

        for path in self.DEFAULT_LOCAL_PATHS:
            expanded = os.path.expanduser(path)
            if os.path.exists(expanded):
                print(f"[{self.name}] Found local dataset at: {expanded}")
                return expanded

        raise FileNotFoundError(
            f"VQA-Med-2021 dataset not found. Please set data_path in DataConfig.\n"
            f"Checked locations:\n" +
            "\n".join(f"  - {p}" for p in self.DEFAULT_LOCAL_PATHS)
        )

    def _load_raw(self) -> Dataset:
        """Load VQA-Med-2021 from local pipe-separated text files."""
        data_path = self._find_local_path()

        questions_file = os.path.join(data_path, "Task1-VQA-2021-TestSet-Questions.txt")
        answers_file = os.path.join(data_path, "Task1-VQA-2021-TestSet-ReferenceAnswers.txt")
        imgs_dir = os.path.join(data_path, "images", "VQA-500-Images")

        if not os.path.exists(questions_file):
            raise FileNotFoundError(f"Questions file not found: {questions_file}")
        if not os.path.exists(answers_file):
            raise FileNotFoundError(f"Answers file not found: {answers_file}")
        if not os.path.exists(imgs_dir):
            raise FileNotFoundError(f"Images directory not found: {imgs_dir}")

        print(f"[{self.name}] Loading from local: {data_path}")

        # Parse questions: image_id|question
        questions = {}
        with open(questions_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|", 1)
                if len(parts) == 2:
                    questions[parts[0].strip()] = parts[1].strip()

        # Parse answers: image_id|answer1|answer2|...
        answers = {}
        with open(answers_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    image_id = parts[0].strip()
                    answer_list = [a.strip() for a in parts[1:] if a.strip()]
                    answers[image_id] = answer_list

        # Build samples
        samples = []
        missing_images = 0

        for image_id, question in questions.items():
            img_path = os.path.join(imgs_dir, f"{image_id}.jpg")

            if not os.path.exists(img_path):
                missing_images += 1
                continue

            answer_list = answers.get(image_id, [])
            if not answer_list:
                continue

            # Primary answer is the first one
            primary_answer = answer_list[0]

            samples.append({
                "image": img_path,
                "question": question,
                "answer": primary_answer,
                "answer_type": "open",  # All questions are open-ended
                "question_id": f"vqamed2021_q_{image_id}",
                "image_id": image_id,
                "dataset_source": self.name,
            })

        if missing_images > 0:
            print(f"[{self.name}] Warning: {missing_images} images not found")

        dataset = Dataset.from_list(samples, features=VQA_FEATURES)
        print(f"[{self.name}] Loaded {len(dataset)} samples (images loaded lazily)")

        return dataset


# Dataset Registry
_DATASET_REGISTRY: Dict[DatasetName, type] = {
    DatasetName.VQA_RAD: VQARADDataset,
    DatasetName.SLAKE: SLAKEDataset,
    DatasetName.VQA_MED_2019: VQAMed2019Dataset,
    DatasetName.VQA_MED_2020: VQAMed2020Dataset,
    DatasetName.VQA_MED_2021: VQAMed2021Dataset,
}


def get_dataset(config: DataConfig) -> BaseVQADataset:
    """Factory function to get the appropriate dataset class.

    Args:
        config: Dataset configuration specifying name, split, and filtering.

    Returns:
        An unloaded dataset instance. Call .load() to fetch data.
    """
    if config.dataset_name not in _DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset: {config.dataset_name}. "
                        f"Available: {list(_DATASET_REGISTRY.keys())}")

    dataset_cls = _DATASET_REGISTRY[config.dataset_name]
    return dataset_cls(config)


def register_dataset(name: DatasetName, dataset_cls: type):
    """Register a new dataset class.

    Args:
        name: Dataset identifier to register under.
        dataset_cls: A BaseVQADataset subclass.
    """
    _DATASET_REGISTRY[name] = dataset_cls


def list_available_datasets() -> List[str]:
    """List all registered dataset names.

    Returns:
        List of dataset name strings (e.g. ["vqa_rad", "slake", ...]).
    """
    return [d.value for d in _DATASET_REGISTRY.keys()]


def train_val_test_split(
    dataset: Dataset,
    val_fraction: float = 0.3,
    seed: int = 42,
) -> tuple:
    """Split a dataset into validation and test subsets.

    Useful for datasets that only ship a single test split (e.g., VQA-Med)
    and you need a held-out validation set for calibration fitting.

    Args:
        dataset: HuggingFace Dataset to split.
        val_fraction: Fraction of data to use for validation (default 0.3).
        seed: Random seed for reproducible splits.

    Returns:
        Tuple of (val_dataset, test_dataset).

    Example::

        dataset = medvlm.load_dataset("vqa_med_2019", split="test", data_path="...")
        val_set, test_set = train_val_test_split(dataset, val_fraction=0.3, seed=42)
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    if not isinstance(dataset, Dataset):
        raise TypeError(
            f"train_val_test_split expects a datasets.Dataset, got {type(dataset).__name__}. "
            "For plain lists, shuffle and slice directly, e.g.:\n"
            "    import random\n"
            "    random.Random(seed).shuffle(examples)\n"
            "    k = int(val_fraction * len(examples))\n"
            "    val, test = examples[:k], examples[k:]"
        )

    shuffled = dataset.shuffle(seed=seed)
    val_size = int(len(shuffled) * val_fraction)

    val_dataset = shuffled.select(range(val_size))
    test_dataset = shuffled.select(range(val_size, len(shuffled)))

    print(f"Split {len(dataset)} samples -> val={len(val_dataset)}, test={len(test_dataset)} "
          f"(val_fraction={val_fraction}, seed={seed})")

    return val_dataset, test_dataset
