import csv
import zipfile
from pathlib import Path

import lightning as L
import molvs
import numpy as np
import requests
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from .utils import UnionFind


class SMILESDataset(Dataset):
    def __init__(
        self,
        smiles_list: list[str],
        tokenizer,
        max_length: int = 512,
        randomize: bool = False,
        canonical: bool = False,
        scaffold_only: bool = False,
    ):
        self.smiles_list = smiles_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.randomize = randomize
        self.canonical = canonical
        self.scaffold_only = scaffold_only

    def __len__(self):
        return len(self.smiles_list)

    def _process_smiles(self, smiles: str) -> str:
        """Process SMILES with optional scaffold extraction, canonicalization and randomization."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return smiles

            # Extract scaffold if scaffold_only mode is enabled
            if self.scaffold_only:
                try:
                    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
                    mol = scaffold  # Use scaffold for further processing
                except Exception:
                    # Keep original molecule if scaffold extraction fails
                    pass

            if self.canonical:
                smiles = Chem.MolToSmiles(mol, canonical=True)
            elif self.randomize:
                smiles = Chem.MolToSmiles(mol, doRandom=True, canonical=False)
            else:
                # If scaffold was extracted but no canonical/randomize, convert to SMILES
                smiles = Chem.MolToSmiles(mol, isomericSmiles=False)

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
        similarity_threshold: float = 0.6,
        scaffold_only: bool = False,
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
        self.similarity_threshold = similarity_threshold
        self.scaffold_only = scaffold_only

        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.collate_fn = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

        self.mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

        self.data_dir.mkdir(exist_ok=True)

    def _standardize_smiles(self, smiles: str) -> str:
        try:
            return molvs.standardize_smiles(smiles)
        except Exception:
            return smiles

    def _download_coconut(self) -> Path:
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
        smiles_file = self.data_dir / f"{self.dataset_type}_all.smi"

        if smiles_file.exists():
            print(f"Loading cached SMILES for {self.dataset_type} dataset")
            with open(smiles_file, "r") as f:
                smiles_list = [line.strip() for line in f]
            return

        if self.dataset_type == "natural":
            filepath = self._download_coconut()

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

        elif self.dataset_type == "general":
            filepath = self.data_dir / "chembl35.txt"

            if not filepath.exists():
                raise FileNotFoundError(
                    f"ChEMBL data file not found at {filepath}. "
                    "Please ensure chembl35.txt is in the data directory."
                )

            # Read SMILES from ChEMBL file (tab-separated: SMILES\tCHEMBL_ID)
            smiles_list = []
            with open(filepath, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:  # Skip empty lines
                        continue

                    parts = line.split("\t")
                    if len(parts) >= 1:  # At least SMILES column
                        smiles = parts[0]  # First column is SMILES
                        if smiles and len(smiles) < 500:
                            smiles_list.append(smiles)

        else:
            raise ValueError(
                f"Unknown dataset type: {self.dataset_type}. Must be 'natural', 'synthetic', or 'general'"
            )

        if self.max_samples:
            smiles_list = smiles_list[: self.max_samples]

        # Save processed SMILES
        smiles_file = self.data_dir / f"{self.dataset_type}_all.smi"
        with open(smiles_file, "w") as f:
            for smiles in tqdm(smiles_list, desc="Standardizing and saving SMILES"):
                smiles = self._standardize_smiles(smiles)
                f.write(smiles + "\n")

        print(f"Prepared {len(smiles_list)} SMILES for {self.dataset_type} dataset")

    def _get_scaffold(self, smiles: str) -> str | None:
        """Extract Bemis-Murcko scaffold from SMILES."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            # Remove chirality to group stereoisomers together
            return Chem.MolToSmiles(scaffold, isomericSmiles=False)
        except Exception:
            return None

    def _get_ecfp4_fingerprint(self, smiles: str):
        """Calculate ECFP4 fingerprint from SMILES using MorganGenerator."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            # ECFP4: radius=2, 2048 bits
            fp = self.mfpgen.GetFingerprint(mol)
            return fp
        except Exception:
            return None

    def scaffold_split(
        self, smiles_list: list[str], val_split: float, test_split: float
    ):
        """Split dataset based on molecular scaffolds.

        This method groups molecules by scaffold, then clusters similar scaffolds
        using ECFP4 fingerprints to prevent similar molecules from appearing in different splits.
        """
        # Extract scaffolds and create group labels
        scaffolds = []
        for smiles in tqdm(smiles_list, desc="Extracting scaffolds"):
            scaffold = self._get_scaffold(smiles)
            if scaffold is None:
                scaffold = smiles  # Use SMILES itself if scaffold extraction fails
            scaffolds.append(scaffold)

        # Map scaffolds to integer group IDs and find representative molecules
        unique_scaffolds = list(set(scaffolds))
        scaffold_to_id = {
            scaffold: idx for idx, scaffold in enumerate(unique_scaffolds)
        }
        scaffold_to_indices = {scaffold: [] for scaffold in unique_scaffolds}
        for idx, scaffold in enumerate(scaffolds):
            scaffold_to_indices[scaffold].append(idx)

        print(f"Found {len(unique_scaffolds)} unique scaffolds")

        # Calculate ECFP4 fingerprints for scaffolds
        print("Calculating ECFP4 fingerprints for scaffolds...")
        scaffold_fps = []
        for scaffold in tqdm(unique_scaffolds, desc="Computing fingerprints"):
            fp = self._get_ecfp4_fingerprint(scaffold)
            if fp is None:
                # Create empty fingerprint for invalid scaffolds
                fp = DataStructs.ExplicitBitVect(2048)
            scaffold_fps.append(fp)

        # Cluster scaffolds using Union-Find
        uf = UnionFind(len(unique_scaffolds))
        n_scaffolds = len(scaffold_fps)

        for i in tqdm(range(n_scaffolds), desc="Finding similar scaffolds"):
            sims = DataStructs.BulkTanimotoSimilarity(
                scaffold_fps[i], scaffold_fps[i + 1 :]
            )
            for j, sim in enumerate(sims):
                if sim >= self.similarity_threshold:
                    uf.union(i, i + j + 1)

        # Get cluster labels
        scaffold_cluster_labels = uf.get_clusters()

        print(
            f"Scaffold clusters: {len(set(scaffold_cluster_labels))} clusters from {len(unique_scaffolds)} scaffolds"
        )

        # Assign cluster labels to all molecules based on their scaffold
        molecule_cluster_labels = np.array(
            [
                scaffold_cluster_labels[scaffold_to_id[scaffold]]
                for scaffold in scaffolds
            ]
        )

        # First split: train vs (val + test)
        test_val_split = val_split + test_split
        gss_train = GroupShuffleSplit(
            n_splits=1, test_size=test_val_split, random_state=42
        )
        train_indices, test_val_indices = next(
            gss_train.split(np.arange(len(smiles_list)), groups=molecule_cluster_labels)
        )

        # Second split: val vs test from the test_val set
        test_val_groups = molecule_cluster_labels[test_val_indices]
        val_ratio = val_split / test_val_split
        gss_test = GroupShuffleSplit(
            n_splits=1, test_size=1 - val_ratio, random_state=42
        )
        val_indices_local, test_indices_local = next(
            gss_test.split(test_val_indices, groups=test_val_groups)
        )

        # Convert local indices back to global indices
        val_indices = test_val_indices[val_indices_local]
        test_indices = test_val_indices[test_indices_local]

        print(
            f"Split sizes: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}"
        )

        return train_indices.tolist(), val_indices.tolist(), test_indices.tolist()

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
                scaffold_only=self.scaffold_only,
            )
            self.val_dataset = SMILESDataset(
                val_smiles,
                self.tokenizer,
                self.max_length,
                randomize=False,  # Don't randomize validation
                canonical=self.canonical,
                scaffold_only=self.scaffold_only,
            )
            self.test_dataset = SMILESDataset(
                test_smiles,
                self.tokenizer,
                self.max_length,
                randomize=False,  # Don't randomize test
                canonical=self.canonical,
                scaffold_only=self.scaffold_only,
            )

            print(
                f"Loaded splits: train={len(train_smiles)}, val={len(val_smiles)}, test={len(test_smiles)}"
            )
            return

        # Load or prepare full dataset
        smiles_file = self.data_dir / f"{self.dataset_type}_all.smi"

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
            scaffold_only=self.scaffold_only,
        )
        self.val_dataset = SMILESDataset(
            val_smiles_selected,
            self.tokenizer,
            self.max_length,
            randomize=False,
            canonical=self.canonical,
            scaffold_only=self.scaffold_only,
        )
        self.test_dataset = SMILESDataset(
            test_smiles_selected,
            self.tokenizer,
            self.max_length,
            randomize=False,
            canonical=self.canonical,
            scaffold_only=self.scaffold_only,
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
