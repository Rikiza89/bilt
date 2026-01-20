# Install
pip install -e .

# Basic usage
from bilt import BILT
model = BILT("weights.pth")
results = model.predict("test.jpg")