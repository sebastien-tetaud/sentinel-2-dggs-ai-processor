# Sentinel-2 DGGS AI Processor


## Installation

1. Clone the repository:

```bash
git clone git@github.com:sebastien-tetaud/sentinel-2-dggs-ai-processor.git
cd sentinel-2-dggs-ai-processor
```

2. Create and activate a conda environment:

```bash
conda create -n eopf python==3.11.7
conda eopf
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your credentials by creating a `.env` file in the root directory with the following content:

```bash
touch .env
```
then:

```
ACCESS_KEY_ID=username
SECRET_ACCESS_KEY=password
```



## dl data
```bash
python download.py --config /mnt/disk/dataset/sentinel-ai-processor/V2/config_20250418_134103.yaml --l1c-csv /mnt/disk/dataset/sentinel-ai-processor/V2/input_l1c.csv --l2a-csv /mnt/disk/dataset/sentinel-ai-processor/V2/output_l2a.csv
```
