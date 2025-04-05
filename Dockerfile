FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt && \
    rm -rf /root/.cache/pip

CMD ["/bin/bash"]
