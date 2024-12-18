# Meta Llama 3 8B Instruct (W/ Decoding CoT)
Implemented from [OptiLLM](https://github.com/codelion/optillm/blob/main/optillm/cot_decoding.py) with the [paper](https://arxiv.org/abs/2402.10200).

Make sure you have a Beam.Cloud account before you start.

Set up your environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
```

Install Beam CLI:

```bash
pip install beam-client
beam configure default --token "your_token_here"
```

> Note: This is a gated Huggingface model and you must request access to it here: https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

Retrieve your HF token from this page: https://huggingface.co/settings/tokens

After your access is granted, make sure to save your Huggingface token on Beam (or via the platform):

```bash
beam secret create HF_TOKEN
```
After you are done you can serve the endpoint:

```bash
beam serve app.py:generate_text
```

The first time around it will take awhile to cache the model. You may need to serve several times. After it has been cached, it will go up within a minute. Caching the model in Volumes in Beam.Cloud does not cost anything as it is now. 

After the endpoint is deployed, you can call it like this:

```sh
curl -X POST 'https://app.beam.cloud/endpoint/id/[ENDPOINT-ID]' \
-H 'Connection: keep-alive' \
-H 'Content-Type: application/json' \
-H 'Authorization: Bearer [AUTH-TOKEN]' \
-d '{
    "messages": [
        {"role": "user", "content": ""Which of the following activities constitute real sector in the economy? 1. Farmers harvesting their crops 2. Textile mills converting raw cotton into fabrics 3. A commercial bank lending money to a trading company 4. A corporate body issuing Rupee Denominated Bonds overseas. Select the correct answer using the code given below: a) 1 and 2 only,b) 2, 3 and 4 only, c) 1, 3 and 4 only, d) 1, 2, 3 and 4""}
    ]
}'
```

If you want to deploy you simply run:

```bash
beam deploy app.py:generate_text
```
