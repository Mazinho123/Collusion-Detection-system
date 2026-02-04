# Deployment Guide

This guide explains how to deploy the Bidder Collusion Detection System.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for cloning from GitHub)

## Installation from GitHub

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/CollusionDetection.git
cd CollusionDetection
```

### 2. Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate

# On macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

Or using setup.py:

```bash
pip install -e .
```

Or using pyproject.toml:

```bash
pip install .
```

### 4. Verify Installation

```bash
python test_demo.py
```

## Running the System

### Quick Start

```bash
python main_pipeline.py
```

### With Custom Configuration

Edit `config.py` to customize parameters:

```python
from config import Config

config = Config(preset='aggressive')  # or 'balanced', 'conservative'
# Customize parameters as needed
```

### Batch Processing

```python
from main_pipeline import run_analysis
import pandas as pd

# Load your auction data
data = pd.read_csv('your_auctions.csv')

# Run analysis
results = run_analysis(data, config)
```

## Docker Deployment (Optional)

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "main_pipeline.py"]
```

Build and run:

```bash
docker build -t collusion-detection .
docker run collusion-detection
```

## Cloud Deployment

### AWS Lambda

```python
import json
from main_pipeline import run_analysis

def lambda_handler(event, context):
    data = json.loads(event['body'])
    results = run_analysis(data)
    return {
        'statusCode': 200,
        'body': json.dumps(results.to_dict())
    }
```

### Google Cloud Functions

```python
def analyze_bids(request):
    from main_pipeline import run_analysis
    
    request_json = request.get_json()
    results = run_analysis(request_json['data'])
    
    return results.to_dict()
```

### Azure Functions

```python
import azure.functions as func
from main_pipeline import run_analysis

def main(req: func.HttpRequest) -> func.HttpResponse:
    data = req.get_json()
    results = run_analysis(data)
    return func.HttpResponse(json.dumps(results.to_dict()))
```

## Environment Variables

Create `.env` file:

```
LOG_LEVEL=INFO
DATA_PATH=./data
OUTPUT_PATH=./results
RANDOM_SEED=42
```

Load in your code:

```python
import os
from dotenv import load_dotenv

load_dotenv()
log_level = os.getenv('LOG_LEVEL', 'INFO')
```

## Production Checklist

- [ ] All tests passing
- [ ] Dependencies pinned to specific versions
- [ ] Environment variables configured
- [ ] Logging properly configured
- [ ] Error handling in place
- [ ] Documentation updated
- [ ] Code reviewed
- [ ] Security scan completed
- [ ] Performance tested
- [ ] Backup strategy defined

## Troubleshooting Deployment

### Module Import Errors

```bash
pip install --upgrade pip
pip install -r requirements.txt --force-reinstall
```

### Memory Issues

Reduce contamination parameter in `config.py`:
```python
config.anomaly_contamination = 0.05  # instead of 0.10
```

### Performance Issues

Enable logging to diagnose:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Support

For deployment issues, check:
- [QUICKSTART.md](QUICKSTART.md) - Quick start guide
- [README.md](README.md) - Full documentation
- [GitHub Issues](https://github.com/yourusername/CollusionDetection/issues)

## Scaling Considerations

- Horizontal: Distribute data processing across multiple workers
- Vertical: Increase computational resources
- Caching: Cache feature engineering results
- Batching: Process data in batches

## Security

- Use environment variables for sensitive config
- Validate all input data
- Implement rate limiting for API endpoints
- Use HTTPS for data transmission
- Keep dependencies updated: `pip install --upgrade -r requirements.txt`

## Monitoring

Set up monitoring for:
- Processing time
- Memory usage
- Error rates
- Model drift
- Data quality

## Rollback Procedure

```bash
# Revert to previous version
git checkout <previous-commit>
pip install -r requirements.txt
python main_pipeline.py
```

## Contact

For deployment support or questions, please open an issue on GitHub.
