# BILT (Because I Like Twice) - A PyTorch-based object detection library -  AGPL-3.0 License.

"""
Basic tests for BILT library.
"""

import pytest
import torch
from pathlib import Path
from PIL import Image
import numpy as np
import tempfile
import shutil

from bilt import BILT
from bilt.utils import parse_bilt_label, validate_dataset_structure


class TestBILT:
    """Test BILT main class."""
    
    def test_init_empty(self):
        """Test initialization without weights."""
        model = BILT()
        assert model.model is None
        assert model.class_names is None
    
    def test_device_selection(self):
        """Test device selection."""
        model_cpu = BILT(device="cpu")
        assert model_cpu.device == torch.device("cpu")
    
    def test_repr(self):
        """Test string representation."""
        model = BILT()
        assert "untrained" in str(model)


class TestUtils:
    """Test utility functions."""
    
    def test_parse_yolo_label(self):
        """Test YOLO label parsing."""
        # Create temporary label file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("0 0.5 0.5 0.4 0.6\n")
            f.write("1 0.3 0.3 0.2 0.2\n")
            label_path = Path(f.name)
        
        try:
            annotations = parse_bilt_label(label_path, 640, 480)
            
            assert len(annotations) == 2
            assert annotations[0]['class_id'] == 0
            assert annotations[1]['class_id'] == 1
            
            # Check bbox format [x_min, y_min, x_max, y_max]
            bbox = annotations[0]['bbox']
            assert len(bbox) == 4
            assert all(isinstance(v, float) for v in bbox)
        finally:
            label_path.unlink()
    
    def test_validate_dataset_structure(self):
        """Test dataset structure validation."""
        # Create temporary dataset structure
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            
            # Create required directories
            (dataset_path / "train" / "images").mkdir(parents=True)
            (dataset_path / "train" / "labels").mkdir(parents=True)
            (dataset_path / "val" / "images").mkdir(parents=True)
            (dataset_path / "val" / "labels").mkdir(parents=True)
            
            # Add dummy image
            img = Image.new('RGB', (100, 100))
            img.save(dataset_path / "train" / "images" / "test.jpg")
            
            valid, message = validate_dataset_structure(dataset_path)
            assert valid
            assert "valid" in message.lower()
    
    def test_validate_invalid_structure(self):
        """Test validation with invalid structure."""
        with tempfile.TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir)
            valid, message = validate_dataset_structure(dataset_path)
            assert not valid


class TestDataset:
    """Test dataset functionality."""
    
    def setup_method(self):
        """Setup test dataset."""
        self.tmpdir = tempfile.mkdtemp()
        self.dataset_path = Path(self.tmpdir)
        
        # Create structure
        images_dir = self.dataset_path / "images"
        labels_dir = self.dataset_path / "labels"
        images_dir.mkdir(parents=True)
        labels_dir.mkdir(parents=True)
        
        # Create test image
        img = Image.new('RGB', (640, 480), color='red')
        img.save(images_dir / "test.jpg")
        
        # Create test label
        with open(labels_dir / "test.txt", 'w') as f:
            f.write("0 0.5 0.5 0.3 0.4\n")
    
    def teardown_method(self):
        """Cleanup test dataset."""
        shutil.rmtree(self.tmpdir)
    
    def test_dataset_loading(self):
        """Test dataset can be loaded."""
        from bilt.dataset import ObjectDetectionDataset, get_transforms
        
        dataset = ObjectDetectionDataset(
            images_dir=self.dataset_path / "images",
            labels_dir=self.dataset_path / "labels",
            transforms=get_transforms(640, training=True),
            input_size=640
        )
        
        assert len(dataset) == 1
        assert dataset.num_classes > 0
        
        # Test __getitem__
        img, target = dataset[0]
        assert isinstance(img, torch.Tensor)
        assert 'boxes' in target
        assert 'labels' in target


class TestInference:
    """Test inference functionality."""
    
    def test_predict_input_types(self):
        """Test different input types for prediction."""
        # This would require a trained model
        # Placeholder test structure
        pass


if __name__ == "__main__":

    pytest.main([__file__, "-v"])

