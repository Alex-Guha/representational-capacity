import os
import json
from safetensors.torch import safe_open
from huggingface_hub import hf_hub_download, errors

os.environ['HF_XET_HIGH_PERFORMANCE'] = '1'


def get_embedding_matrix(repo, device=0, save_to_dir='./models/embd_only'):
    model_name = repo.split('/')[-1]
    os.makedirs(os.path.join(save_to_dir, model_name), exist_ok=True)
    try:
        hf_hub_download(
            repo_id=repo, filename="model.safetensors.index.json", local_dir=os.path.join(save_to_dir, model_name))

        with open(os.path.join(save_to_dir, model_name, 'model.safetensors.index.json')) as f:
            model_index = json.load(f)

        postfix = 'model.embed_tokens.weight'
        try:
            model_index['weight_map'][postfix]
        except KeyError:
            postfix = 'language_model.model.embed_tokens.weight'

        hf_hub_download(
            repo_id=repo, filename=model_index['weight_map'][postfix], local_dir=os.path.join(save_to_dir, model_name))

        with safe_open(os.path.join(save_to_dir, model_name, model_index['weight_map'][postfix]), framework='pt') as f:
            return f.get_tensor(postfix).to(device).T

    except (errors.HfHubHTTPError, errors.RemoteEntryNotFoundError) as e:
        try:
            hf_hub_download(
                repo_id=repo, filename='model.safetensors', local_dir=os.path.join(save_to_dir, model_name))

            with safe_open(os.path.join(save_to_dir, model_name, 'model.safetensors'), framework='pt') as f:
                return f.get_tensor('model.embed_tokens.weight').to(device).T

        except (errors.HfHubHTTPError, errors.RemoteEntryNotFoundError) as e:
            print(
                f"Error downloading or loading model {model_name} from repo {repo}: {e}")
            return None


def get_unembedding_matrix(repo, device=0, save_to_dir='./models/embd_only'):
    model_name = repo.split('/')[-1]
    os.makedirs(os.path.join(save_to_dir, model_name), exist_ok=True)
    try:
        hf_hub_download(repo_id=repo, filename="model.safetensors.index.json",
                        local_dir=os.path.join(save_to_dir, model_name))

        with open(os.path.join(save_to_dir, model_name, 'model.safetensors.index.json')) as f:
            model_index = json.load(f)

        postfix = 'lm_head.weight'
        try:
            model_index['weight_map'][postfix]
        except KeyError:
            postfix = 'language_model.lm_head.weight'

        hf_hub_download(repo_id=repo, filename=model_index['weight_map']
                        [postfix], local_dir=os.path.join(save_to_dir, model_name))

        with safe_open(os.path.join(save_to_dir, model_name, model_index['weight_map'][postfix]), framework='pt') as f:
            return f.get_tensor(postfix).to(device).T

    except (errors.HfHubHTTPError, errors.RemoteEntryNotFoundError) as e:
        try:
            hf_hub_download(repo_id=repo, filename='model.safetensors', local_dir=os.path.join(save_to_dir, model_name))

            with safe_open(os.path.join(save_to_dir, model_name, 'model.safetensors'), framework='pt') as f:
                return f.get_tensor('lm_head.weight').to(device).T

        except (errors.HfHubHTTPError, errors.RemoteEntryNotFoundError, Exception) as e:
            print(f"Error downloading or loading model {model_name} from repo {repo}: {e}")
            return None
    except KeyError as e:
        print(f"No lm_head.weight found")
        return None
