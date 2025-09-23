import csv
import zipfile
from collections import defaultdict
from pathlib import Path

import lightning as L
import requests
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


class SMILESDataset(Dataset):
    def __init__(
        self,
        smiles_list: list[str],
        tokenizer,
        max_length: int = 512,
        randomize: bool = False,
        canonical: bool = False,
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.randomize = randomize
        self.canonical = canonical

    def __len__(self):
        return len(self.smiles_list)

    def _process_smiles(self, smiles: str) -> str:
        """Process SMILES with optional canonicalization and randomization."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles

            if self.canonical:
                smiles = Chem.MolToSmiles(mol, canonical=True)
            elif self.randomize:
                smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)

            return smiles
        except Exception:
            return smiles

    def __getitem__(self, idx):
        smiles = self.smiles_list[idx]

        # Process SMILES if needed
        smiles = self._process_smiles(smiles)

        # Tokenize and return the full encoding dict (DataCollator needs this format)
        encoding = self.tokenizer(
            smiles,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding.input_ids.squeeze(),
            "attention_mask": encoding.attention_mask.squeeze(),
        }


class SMILESDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        tokenizer_name: str = "kohbanye/SmilesTokenizer_PubChem_1M",
        batch_size: int = 32,
        max_length: int = 512,
        num_workers: int = 4,
        val_split: float = 0.1,
        test_split: float = 0.1,
        dataset_type: str = "natural",  # "natural" or "synthetic"
        max_samples: int | None = None,
        randomize: bool = False,
        canonical: bool = False,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.tokenizer_name = tokenizer_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.num_workers = num_workers
        self.val_split = val_split
        self.test_split = test_split
        self.dataset_type = dataset_type
        self.max_samples = max_samples
        self.randomize = randomize
        self.canonical = canonical

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        self.data_dir.mkdir(exist_ok=True)

    def download_coconut(self) -> Path:
        coconut_path = self.data_dir / "coconut.csv"

        if coconut_path.exists():
            return coconut_path

        print("Downloading COCONUT dataset...")
        url = "https://coconut.s3.uni-jena.de/prod/downloads/2025-08/coconut_csv-08-2025.zip"
        response = requests.get(url, stream=True)
        response.raise_for_status()

        zip_path = self.data_dir / "coconut.zip"
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(self.data_dir)

        zip_path.unlink()
        extracted_csv = self.data_dir / "coconut_csv-08-2025.csv"
        extracted_csv.rename(coconut_path)

        return coconut_path

    def prepare_data(self):
        if self.dataset_type == "natural":
            filepath = self.download_coconut()

            # Extract SMILES from CSV
            csv.field_size_limit(csv.field_size_limit() * 10)
            smiles_list = []
            with open(filepath, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "canonical_smiles" in row and row["canonical_smiles"]:
                        smiles = row["canonical_smiles"].strip()
                        if smiles and len(smiles) < 500:
                            smiles_list.append(smiles)

        elif self.dataset_type == "synthetic":
            filepath = self.data_dir / "zinc22.txt"

            # Read SMILES from file
            smiles_list = []
            with open(filepath, "r") as f:
                for line_num, line in enumerate(f):
                    line = line.strip()
                    if not line or line_num == 0:  # Skip empty lines and header
                        continue

                    parts = line.split()
                    if len(parts) >= 3:  # Ensure we have tranche, zincid, and SMILES
                        smiles = parts[2]  # Third column is SMILES
                        if smiles and len(smiles) < 500:
                            smiles_list.append(smiles)

        else:
            raise ValueError(f"Unknown dataset type: {self.dataset_type}")

        if self.max_samples:
            smiles_list = smiles_list[: self.max_samples]

        # Save processed SMILES
        smiles_file = self.data_dir / f"{self.dataset_type}_smiles.txt"
        with open(smiles_file, "w") as f:
            for smiles in smiles_list:
                f.write(smiles + "\n")

        print(f"Prepared {len(smiles_list)} SMILES for {self.dataset_type} dataset")

    def get_scaffold(self, smiles: str) -> str | None:
        """Extract Bemis-Murcko scaffold from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            return Chem.MolToSmiles(scaffold)
        except Exception:
            return None

    def scaffold_split(
        self, smiles_list: list[str], val_split: float, test_split: float
    ):
        """Split dataset based on molecular scaffolds."""
        # Group molecules by scaffold
        scaffold_to_indices = defaultdict(list)
        for idx, smiles in enumerate(tqdm(smiles_list, desc="Extracting scaffolds")):
            scaffold = self.get_scaffold(smiles)
            if scaffold is None:
                scaffold = smiles  # Use SMILES itself if scaffold extraction fails
            scaffold_to_indices[scaffold].append(idx)

        # Sort scaffolds by group size (largest first)
        scaffold_groups = sorted(scaffold_to_indices.values(), key=len, reverse=True)

        # Calculate split sizes
        total_size = len(smiles_list)
        val_size = int(total_size * val_split)
        test_size = int(total_size * test_split)

        # Greedily assign scaffolds to splits
        train_indices, val_indices, test_indices = [], [], []

        for group_indices in scaffold_groups:
            if len(val_indices) < val_size:
                val_indices.extend(group_indices)
            elif len(test_indices) < test_size:
                test_indices.extend(group_indices)
            else:
                train_indices.extend(group_indices)

        return train_indices, val_indices, test_indices

    def setup(self, stage: str | None = None):
        # Check for cached splits
        train_file = self.data_dir / f"{self.dataset_type}_train.smi"
        val_file = self.data_dir / f"{self.dataset_type}_val.smi"
        test_file = self.data_dir / f"{self.dataset_type}_test.smi"

        if train_file.exists() and val_file.exists() and test_file.exists():
            print(f"Loading cached splits for {self.dataset_type} dataset")
            with open(train_file, "r") as f:
                train_smiles = [line.strip() for line in f]
            with open(val_file, "r") as f:
                val_smiles = [line.strip() for line in f]
            with open(test_file, "r") as f:
                test_smiles = [line.strip() for line in f]

            # Create datasets from cached splits
            self.train_dataset = SMILESDataset(
                train_smiles,
                self.tokenizer,
                self.max_length,
                randomize=self.randomize,
                canonical=self.canonical,
            )
            self.val_dataset = SMILESDataset(
                val_smiles,
                self.tokenizer,
                self.max_length,
                randomize=False,  # Don't randomize validation
                canonical=self.canonical,
            )
            self.test_dataset = SMILESDataset(
                test_smiles,
                self.tokenizer,
                self.max_length,
                randomize=False,  # Don't randomize test
                canonical=self.canonical,
            )

            print(
                f"Loaded splits: train={len(train_smiles)}, val={len(val_smiles)}, test={len(test_smiles)}"
            )
            return

        # Load or prepare full dataset
        smiles_file = self.data_dir / f"{self.dataset_type}_smiles.txt"

        if not smiles_file.exists():
            self.prepare_data()

        with open(smiles_file, "r") as f:
            smiles_list = [line.strip() for line in f]

        if self.max_samples:
            smiles_list = smiles_list[: self.max_samples]

        # Perform scaffold split
        train_indices, val_indices, test_indices = self.scaffold_split(
            smiles_list, self.val_split, self.test_split
        )

        # Save split SMILES to files
        train_smiles = [smiles_list[i] for i in train_indices]
        val_smiles = [smiles_list[i] for i in val_indices]
        test_smiles = [smiles_list[i] for i in test_indices]

        with open(train_file, "w") as f:
            for smiles in train_smiles:
                f.write(smiles + "\n")
        with open(val_file, "w") as f:
            for smiles in val_smiles:
                f.write(smiles + "\n")
        with open(test_file, "w") as f:
            for smiles in test_smiles:
                f.write(smiles + "\n")

        print(
            f"Saved splits: train={len(train_smiles)}, val={len(val_smiles)}, test={len(test_smiles)}"
        )

        train_smiles_selected = [smiles_list[i] for i in train_indices]
        val_smiles_selected = [smiles_list[i] for i in val_indices]
        test_smiles_selected = [smiles_list[i] for i in test_indices]

        self.train_dataset = SMILESDataset(
            train_smiles_selected,
            self.tokenizer,
            self.max_length,
            randomize=self.randomize,
            canonical=self.canonical,
        )
        self.val_dataset = SMILESDataset(
            val_smiles_selected,
            self.tokenizer,
            self.max_length,
            randomize=False,
            canonical=self.canonical,
        )
        self.test_dataset = SMILESDataset(
            test_smiles_selected,
            self.tokenizer,
            self.max_length,
            randomize=False,
            canonical=self.canonical,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )
