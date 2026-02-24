import torch, numpy as np
import torch.nn.functional as F
import clip
# from sentence_transformers import SentenceTransformer, util
# from sklearn.preprocessing import binarize
#
# def build_adj_matrix_with_sbert(categories, model_name='all-MiniLM-L6-v2', similarity_threshold=0.5):
#
#     print(f'Loading SentenceTRansformer mode: {model_name}....')
#     model = SentenceTransformer(model_name)
#
#     processed_categories = [item.replace('_', ' ') for item in categories]
#     embeddings = model.encode(processed_categories, convert_to_tensor=True)
#
#     cosine_scores = util.cos_sim(embeddings, embeddings)
#
#     adj_matrix_numpy = cosine_scores.cpu().numpy()
#
#     adj_matrix_binarized = binarize(adj_matrix_numpy, threshold=similarity_threshold)
#
#     np.fill_diagonal(adj_matrix_binarized, 1)
#
#     return torch.from_numpy(adj_matrix_binarized)
#
#
#
# def generate_descriptive_features(clip_pretrained, categories, descriptions, device='cuda'):
#
#     # print('Loading CLIP model')
#     # clip_pretrained, _ = clip.load("ViT-B/32", device='cuda', jit=False, download_root='/home/dgg/haiyang/code/ckpt/clip')
#
#     prompts = [descriptions[item] for item in categories]
#     print('Encoding descriptive prompts')
#
#     with torch.no_grad():
#         tokenized_prompts = clip.tokenize(prompts).to(device)
#         text_features = clip_pretrained.encode_text(tokenized_prompts)
#
#         text_features /= text_features.norm(dim=-1, keepdim=True)
#
#     return text_features

def list_2_tensor(lst, key, tensor: torch.Tensor):
    values = []

    for dct in lst:
        values.append(dct[key])
    if isinstance(values[0], (np.ndarray, list)):
        rst = np.stack(values, axis=0)
    elif isinstance(values[0], torch.Tensor):
        rst = torch.stack(values, dim=0)
    else:
        raise NotImplementedError
    
    return tensor.new_tensor(rst)


def get_rotation_matrix(tensor):
    assert tensor.shape[-1] == 4

    tensor = F.normalize(tensor, dim=-1)
    mat1 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat1[..., 0, 0] = tensor[..., 0]
    mat1[..., 0, 1] = - tensor[..., 1]
    mat1[..., 0, 2] = - tensor[..., 2]
    mat1[..., 0, 3] = - tensor[..., 3]
    
    mat1[..., 1, 0] = tensor[..., 1]
    mat1[..., 1, 1] = tensor[..., 0]
    mat1[..., 1, 2] = - tensor[..., 3]
    mat1[..., 1, 3] = tensor[..., 2]

    mat1[..., 2, 0] = tensor[..., 2]
    mat1[..., 2, 1] = tensor[..., 3]
    mat1[..., 2, 2] = tensor[..., 0]
    mat1[..., 2, 3] = - tensor[..., 1]

    mat1[..., 3, 0] = tensor[..., 3]
    mat1[..., 3, 1] = - tensor[..., 2]
    mat1[..., 3, 2] = tensor[..., 1]
    mat1[..., 3, 3] = tensor[..., 0]

    mat2 = torch.zeros(*tensor.shape[:-1], 4, 4, dtype=tensor.dtype, device=tensor.device)
    mat2[..., 0, 0] = tensor[..., 0]
    mat2[..., 0, 1] = - tensor[..., 1]
    mat2[..., 0, 2] = - tensor[..., 2]
    mat2[..., 0, 3] = - tensor[..., 3]
    
    mat2[..., 1, 0] = tensor[..., 1]
    mat2[..., 1, 1] = tensor[..., 0]
    mat2[..., 1, 2] = tensor[..., 3]
    mat2[..., 1, 3] = - tensor[..., 2]

    mat2[..., 2, 0] = tensor[..., 2]
    mat2[..., 2, 1] = - tensor[..., 3]
    mat2[..., 2, 2] = tensor[..., 0]
    mat2[..., 2, 3] = tensor[..., 1]

    mat2[..., 3, 0] = tensor[..., 3]
    mat2[..., 3, 1] = tensor[..., 2]
    mat2[..., 3, 2] = - tensor[..., 1]
    mat2[..., 3, 3] = tensor[..., 0]

    mat2 = torch.conj(mat2).transpose(-1, -2)
    
    mat = torch.matmul(mat1, mat2)
    return mat[..., 1:, 1:]
