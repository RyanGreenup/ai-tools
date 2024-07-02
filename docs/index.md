# AI Tools

## Installing

First install ollama and pull some models like `phi3` `phi3:mini-128k` and `codestral`, then use Poetry to initialize and run the project:

```sh
git clone https://github.com/RyanGreenup/ai-tools
cd ai-tools
poetry install
poetry run src/main.py -n ~/Notes/slipbox live-search
# Or from any directory
dir=$(pwd)
cd /tmp/
poetry run -C "${dir}" /home/ryan/.local/scripts/python/ai-tools/src/main.py  --help
```

### Ollama

Configure ollama by reading the docs, something like this should work with docker:


```yaml
version: '3.9'
services:
    ollama:
        restart: unless-stopped
        image: ollama/ollama
        container_name: ollama
        ports:
            - '11434:11434'
        volumes:
            - './data:/root/.ollama'
        deploy:
            resources:
              reservations:
                devices:
                  - driver: nvidia
                    device_ids: ['0', '1']
                 #  device_ids: ['0']
                    capabilities: [gpu]

      # Optionally include Open-WebUI
    open-webui:
        image: 'ghcr.io/open-webui/open-webui:main'
        restart: always
        container_name: open-webui
        volumes:
            - './data_open-webui:/app/backend/data'
# Uncomment this for external ollama
#       environment:
#           - 'OLLAMA_BASE_URL=https://example.com'
# Comment or uncomment this for locally hosted ollama
        extra_hosts:
            - 'host.docker.internal:host-gateway'
        ports:
            - '3000:8080'
        depends_on:
            - ollama
```


## Usage

### RAG

#### Personal Content

First pull down a long context model:

```sh
zi ollama
docker compose exec -it ollama bash
ollama pull phi3:mini-128k
```

Then run the RAG model:

```sh
src/main.py \
    -c "phi3:mini-128k" \
        rag             \
            --n-docs=10            \
            --context-length 16000 \
            -E
```


#### Docs

```bash
cd /tmp
git clone https://github.com/pytorch/pytorch --depth=1
~/.local/scripts/python/ai-tools/src/main.py -n pytorch/docs -c "phi3:mini-128k" rag --n-docs=10 -E --context-length 16000
```

Provide the following input:

```markdown
# System
You are a helpful AI assistant
# User
Show how to use a dataloader in pytorch by writing a minimum working example.
```

And the output should look something like this:


> To demonstrate a minimum working example of using `DataLoader` in PyTorch, let's consider loading data from a custom dataset that simulates simple numerical samples (e.g., sequences) and uses DataLoader to batch these into mini-batches for efficient processing during training or inference tasks. This example will showcase how to create a basic DataLoader pipeline by combining a PyTorch `Dataset` object with the DataLoader itself, ensuring that each sample is preprocessed before being fed into your model's forward method.
>
> First, ensure you have PyTorch installed:
>
> ```python
> pip install torch torchvision  # if not already available
> ```
>
> Now, let's create a simple dataset and DataLoader example:
>
> ```python
> import torch
> from torch.utils.data import Dataset, DataLoader
> import numpy as np
>
> # Define a custom Dataset class that simulates numerical samples for each data point
> class CustomDataset(Dataset):
>     def __init__(self, num_samples=1000):
>         self.num_samples = num_samples
>
>     def __len__(self):
>         return self.num_samples
>
>     def __getitem__(self, idx):
>         # Simulate a dataset where each sample is a sequence of numbers; for simplicity, let's generate random sequences
>         seq_length = 10
>         data = np.random.randint(0, 256, size=(self.num_samples, seq_length)).astype('uint8') # Random values between 0 and 255 for each sample sequence
>         return torch.tensor(data, dtype=torch.float32) / 255.0  # Normalize pixel values (for a simple example, assuming binary images)
>
> # Instantiate the dataset with a specified number of samples
> dataset = CustomDataset()
>
> # Define batch size and shuffle option for DataLoader
> batch_size = 32
> shuffle = True
>
> # Create a DataLoader that takes in your custom Dataset object, specifying batch size and whether to shuffle the data
> dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
>
> # Iterate through the DataLoader to access mini-batches of data
> for i, samples in enumerate(dataloader):
>     print("Batch {}".format(i+1))
>
>     # Each 'samples' is a tensor containing batch_size elements. For this simple example, each element is a 2D float32 tensor of shape (batch_size, seq_length).
>     for sample in samples:
>         print('Sample shape:', sample.shape)  # Example output: torch.Size([32, 10])
>
>         # Assuming we have a simple model that expects normalized values as input and doesn't require preprocessing before forward pass
>         # batch_input = dataloader[i % len(dataloader)].unsqueeze(-1)  # Unsqueeze to make the sample dimension available for neural network models expecting a single channel (like image data).
>
>         # Forward pass through your model. In real applications, this would be replaced with actual model logic.
>         output = self.model(batch_input)  # This is just placeholder code. Replace it with your model's forward function.
>         print('Model output shape:', output.shape)
>
>         # Here you would typically have some loss computation and backward pass but we'll skip that for this basic example.
> ```
>
> This script demonstrates how to use PyTorch's `DataLoader` with a custom dataset, showing its simplicity and utility in batching data for model input, crucial for efficient neural network training or inference operations. Note that real-world datasets would require additional steps such as preprocessing (e.g., image transformations) before using the DataLoader.
