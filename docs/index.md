# AI Tools

## Installing

TODO, it's managed by poetry I guess?

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

